"""NASA Science Discovery Engine (SDE) search tool.

This module implements ``sde_search`` — the SDE MCP search contract. It retrieves
SDE-indexed content, normalizes source-specific records into a stable document
contract, and applies deterministic citation normalization to every returned
document.

Design notes:
- The tool executes a validated SDE Search API request against the cross-source
  ``generic`` endpoint or a caller-selected source endpoint.
- Citation normalization runs automatically for every returned document, adding four
  fields (``citation_type``, ``citation_value``, ``fallback_used``, ``citation_status``)
  selected by a fixed identifier hierarchy (DOI → pds_lid → ivo_id → bps_osdr_id →
  url → missing). There is no separate citation-lookup operation: the SDE Search API
  has no lookup-by-identifier endpoint, so citations are only produced from search
  results, where every field the hierarchy needs is already present.
- The tool never fabricates results or identifiers. Upstream and validation failures
  are returned as a structured payload (``success=false``) rather than raised past the
  MCP boundary, so the agent-facing contract stays stable and actionable.
- Responses are size-bounded end to end. ``full_text`` is unbounded per document and both
  the SDE API and the MCP host cap a response at 6 MiB, so a single page can exceed the
  limit. The tool emits ``structuredContent`` only (see ``as_function``) instead of the
  duplicated payload FastMCP produces by default, refetches an oversized window in smaller
  aligned pages, and — only if it still does not fit — drops ``full_text`` from the largest
  documents. Documents are never silently dropped.

Boundaries: summarization, query rewriting, acronym expansion, ranking, and internal
registry operations live in the agent/application layer, not in this tool.
"""

import json
import os
import re
from typing import Literal, NamedTuple

import httpx
from akd._base import InputSchema, OutputSchema
from akd.tools import BaseTool, BaseToolConfig
from fastmcp.tools.tool import ToolResult
from loguru import logger
from pydantic import BaseModel, Field, model_validator

from akd_ext.mcp import mcp_tool
from akd_ext.structures import (
    SDECitationStatus,
    SDECitationType,
    SDEErrorType,
    SDESearchEndpoint,
    SDESearchType,
)

# Maps the caller-facing endpoint enum to the SDE Search API path.
ENDPOINT_PATHS: dict[SDESearchEndpoint, str] = {
    SDESearchEndpoint.GENERIC: "/api/search",
    SDESearchEndpoint.WEB: "/api/web/search",
    SDESearchEndpoint.CMR: "/api/cmr/search",
    SDESearchEndpoint.PDS3: "/api/pds3/search",
    SDESearchEndpoint.PDS4: "/api/pds4/search",
    SDESearchEndpoint.SPASE: "/api/spase/search",
    SDESearchEndpoint.GCN: "/api/gcn/search",
    SDESearchEndpoint.HEK: "/api/hek/search",
    SDESearchEndpoint.NAVO: "/api/navo/search",
    SDESearchEndpoint.OSDR: "/api/osdr/search",
    SDESearchEndpoint.CODE: "/api/code/search",
}

# Statuses worth one immediate replay before giving up or splitting the window.
_RETRY_ONCE_STATUSES = frozenset({500, 502, 503})

# Sentinel status for "HTTP 200 but the body did not parse" — a response truncated in
# transit. Not a real status code, so it cannot collide with one.
_TRUNCATED_RESPONSE = -1

# Failures that mean "this window was too large to deliver". Both the SDE API and the MCP
# host cap a response at 6 MiB: the API answers 502 with `{"message": "Internal server
# error"}`, and a partially delivered body fails to parse. Either way the same window
# succeeds when fetched in smaller pieces.
_SPLIT_STATUSES = frozenset({502, _TRUNCATED_RESPONSE})

class _ErrorPolicy(NamedTuple):
    """How one upstream status maps onto the tool's failure contract.

    ``message`` is a template over ``{status}`` and ``{body}``.
    """

    error_type: SDEErrorType
    message: str
    next_action: str
    retryable: bool


_ERROR_POLICIES: dict[int, _ErrorPolicy] = {
    400: _ErrorPolicy(SDEErrorType.VALIDATION_ERROR, "SDE API rejected the request (HTTP {status}): {body}",
                      "Correct the request; do not retry unchanged.", False),
    422: _ErrorPolicy(SDEErrorType.VALIDATION_ERROR, "SDE API schema validation failed (HTTP {status}): {body}",
                      "Correct the request schema.", False),
    500: _ErrorPolicy(SDEErrorType.UPSTREAM_ERROR, "SDE API internal error (HTTP {status}): {body}",
                      "Report the upstream failure.", False),
    502: _ErrorPolicy(SDEErrorType.RETRYABLE_ERROR, "SDE API bad gateway (HTTP {status}): {body}",
                      "Retry with a smaller page_size; the response likely exceeded the upstream payload limit.", True),
    503: _ErrorPolicy(SDEErrorType.RETRYABLE_ERROR, "SDE API unavailable (HTTP {status}): {body}",
                      "Report service unavailability; retry later.", True),
}

# A 422 naming vectorization is recoverable by rerunning as keyword search, so it overrides
# the generic 422 policy above.
_VECTORIZATION_POLICY = _ErrorPolicy(SDEErrorType.RETRYABLE_ERROR, "Vectorization failed (HTTP {status}): {body}",
                                     "Retry using keyword search (search_type='keyword').", True)


def _unexpected_status_policy(status: int) -> _ErrorPolicy:
    """Fallback for any 4xx/5xx the contract does not name explicitly."""
    return _ErrorPolicy(SDEErrorType.UPSTREAM_ERROR, "SDE API returned HTTP {status}: {body}",
                        "Report the upstream failure.", status >= 500)


# A DOI is `10.<registrant>/<suffix>`, optionally wrapped in a URL/`doi:` prefix.
# The suffix stops at whitespace or a `;`/`,` delimiter — some SDE persistent_id
# fields pack multiple DOIs as `10.x/a;10.y/b`, and only the first is selected.
_DOI_CORE = re.compile(r"10\.\d{4,9}/[^\s;,]+", re.IGNORECASE)


def _extract_doi(value: str | None) -> str | None:
    """Return a bare DOI string if ``value`` contains a valid DOI, else ``None``.

    Recognizes DOIs wrapped as ``https://doi.org/...``, ``doi:...``, or bare, and
    selects the first when a field packs several. A non-DOI persistent identifier
    must never be classified as a DOI, so this only matches the ``10.xxxx/...`` shape.
    """
    if not value or not isinstance(value, str):
        return None
    match = _DOI_CORE.search(value.strip())
    if not match:
        return None
    # Trim trailing punctuation that commonly rides along in free-text fields.
    return match.group(0).rstrip(".,;)")


class SDESearchFilters(BaseModel):
    """Supported SDE search filters.

    OR semantics apply within each list; AND semantics apply between groups.
    Only these four filter fields are supported by the contract.
    """

    division: list[str] | None = Field(
        default=None,
        description="Filter by NASA SMD division (e.g. 'Astrophysics', 'Planetary Science').",
    )
    document_type: list[str] | None = Field(
        default=None,
        description="Filter by document type (e.g. 'Data', 'Documentation', 'Software and Tools').",
    )
    collection_name: list[str] | None = Field(
        default=None, description="Filter by human-readable collection name."
    )
    collection_key: list[str] | None = Field(
        default=None, description="Filter by internal collection key."
    )

    def to_api_payload(self) -> dict:
        """Serialize only the populated filter groups for the API request body."""
        return {k: v for k, v in self.model_dump().items() if v}


class NormalizedDocument(BaseModel):
    """A single SDE document normalized into the stable contract.

    Source-specific fields remain available in ``raw_response``; this model holds
    the fields common across index types plus the four citation fields that are
    added deterministically to every document.
    """

    id: str = Field(default="", description="Normalized document identifier.")
    score: float = Field(default=0.0, description="Relevance score from the search engine.")
    index: str = Field(default="", description="Source index the record came from.")
    title: str = Field(default="", description="Document or dataset title.")
    url: str = Field(default="", description="Direct URL to access the document.")
    division: str = Field(default="", description="NASA SMD division.")
    document_type: str = Field(default="", description="Document type.")
    collection_name: str = Field(default="", description="Collection name.")
    collection_key: str = Field(default="", description="Internal collection key.")
    full_text: str = Field(
        default="",
        description=(
            "Full text or abstract, when available. Never truncated; it is dropped whole "
            "(and flagged by full_text_omitted) only when the response would otherwise "
            "exceed its size budget."
        ),
    )
    full_text_omitted: bool = Field(
        default=False,
        description="True when full_text was dropped to keep the response within its size budget.",
    )
    full_text_chars: int = Field(
        default=0,
        description="Length of the dropped full_text; 0 unless full_text_omitted is true.",
    )
    data_product_desc: str = Field(default="", description="Data product description, when available.")
    relevant_content: str = Field(default="", description="Most relevant snippet for the query.")
    highlights: list[str] = Field(default_factory=list, description="Highlighted matching fragments.")
    persistent_id: str = Field(default="", description="Persistent identifier (may hold a DOI).")
    pds_lid: str = Field(default="", description="PDS logical identifier, when available.")
    ivo_id: str = Field(default="", description="IVOA identifier, when available.")
    bps_osdr_id: str = Field(default="", description="BPS/OSDR identifier, when available.")

    # Citation fields — added deterministically to every document.
    citation_type: SDECitationType = Field(
        default=SDECitationType.MISSING, description="Identifier type selected by the hierarchy."
    )
    citation_value: str = Field(default="", description="Selected DOI, identifier, or URL.")
    fallback_used: bool = Field(
        default=False, description="True when a lower-priority identifier or URL was selected."
    )
    citation_status: SDECitationStatus = Field(
        default=SDECitationStatus.MISSING, description="Availability/quality of the citation."
    )


def normalize_citation(doc: NormalizedDocument) -> None:
    """Apply the deterministic citation hierarchy to ``doc`` in place.

    Hierarchy (first valid, non-empty value wins):
      1. DOI (from ``persistent_id`` or another DOI-bearing field)
      2. ``pds_lid``
      3. ``ivo_id``
      4. ``bps_osdr_id``
      5. external ``url``
      6. missing

    A DOI yields ``complete``/``fallback_used=False``; a lower-priority value
    yields ``fallback``/``fallback_used=True``; no value yields ``missing``.
    Identifiers are never fabricated or inferred.
    """
    doi = _extract_doi(doc.persistent_id)
    if doi:
        doc.citation_type = SDECitationType.DOI
        doc.citation_value = doi
        doc.citation_status = SDECitationStatus.COMPLETE
        doc.fallback_used = False
        return

    for value, ctype in (
        (doc.pds_lid, SDECitationType.PDS_LID),
        (doc.ivo_id, SDECitationType.IVO_ID),
        (doc.bps_osdr_id, SDECitationType.BPS_OSDR_ID),
        (doc.url, SDECitationType.URL),
    ):
        if value:
            doc.citation_type = ctype
            doc.citation_value = value
            doc.citation_status = SDECitationStatus.FALLBACK
            doc.fallback_used = True
            return

    doc.citation_type = SDECitationType.MISSING
    doc.citation_value = ""
    doc.citation_status = SDECitationStatus.MISSING
    doc.fallback_used = False


class SDESearchToolConfig(BaseToolConfig):
    """Instance-time configuration for the SDE search tool."""

    name: str = Field(default="sde_search", description="MCP tool name.")
    base_url: str = Field(
        default=os.getenv(
            "SDE_BASE_URL", "https://science.data.nasa.gov/science-discovery-engine"
        ),
        description="Base URL for the SDE Search API (endpoint paths like /api/search are appended).",
    )
    timeout: float = Field(default=30.0, description="HTTP request timeout in seconds.")
    max_response_bytes: int = Field(
        default=5_000_000,
        description=(
            "Size budget for the serialized document payload. Both the SDE API and the MCP "
            "host cap a response at 6 MiB; this leaves headroom under that. When the payload "
            "is larger, full_text is dropped from the biggest documents until it fits. "
            "0 disables the guard."
        ),
    )


class SDESearchToolInputSchema(InputSchema):
    """Request for an ``sde_search`` query."""

    search_term: str = Field(
        ...,
        min_length=1,
        description="Search query. Must not be blank.",
    )
    endpoint: SDESearchEndpoint = Field(
        default=SDESearchEndpoint.GENERIC,
        description="Search endpoint. 'generic' is cross-source; other values target one source.",
    )
    search_type: SDESearchType = Field(
        default=SDESearchType.HYBRID,
        description="Retrieval strategy: 'hybrid' (vector+keyword), 'keyword', or 'vector'.",
    )
    page: int = Field(default=1, ge=1, description="1-based page number.")
    page_size: int = Field(default=10, ge=1, le=100, description="Results per page (1-100).")
    include_aggregations: bool = Field(
        default=False, description="Request aggregation metadata when available."
    )
    include_raw_documents: bool = Field(
        default=False,
        description=(
            "Include the verbatim upstream document array in raw_response. Off by default; "
            "normalized_documents already contains every document."
        ),
    )
    filters: SDESearchFilters | None = Field(
        default=None, description="Optional result filters (OR within a list, AND across groups)."
    )

    @model_validator(mode="after")
    def _search_term_not_blank(self) -> "SDESearchToolInputSchema":
        if not self.search_term.strip():
            raise ValueError("'search_term' must not be blank.")
        return self


class SDESearchToolOutputSchema(OutputSchema):
    """Response for an ``sde_search`` query.

    A single schema covers both the success and failure contracts so the MCP
    surface stays stable. ``success`` tells the agent which fields are populated;
    ``agent_instruction`` is always present.
    """

    success: bool = Field(..., description="Whether the search succeeded.")
    agent_instruction: str = Field(..., description="Guidance for the agent on how to use this result.")

    # --- Success ---
    endpoint_used: str = Field(default="", description="Endpoint value the search executed against.")
    raw_response: dict = Field(
        default_factory=dict,
        description=(
            "Upstream response metadata. The document array and keys duplicated at the top "
            "level (pagination, aggregations) are pruned unless include_raw_documents=True."
        ),
    )
    normalized_documents: list[NormalizedDocument] = Field(
        default_factory=list, description="Documents normalized into the stable contract."
    )
    pagination: dict = Field(default_factory=dict, description="Pagination metadata (page, page_size, total).")
    aggregations: dict = Field(default_factory=dict, description="Aggregation metadata when requested/available.")

    # --- Failure ---
    error_type: SDEErrorType | None = Field(default=None, description="Structured error category.")
    message: str = Field(default="", description="Human-readable error message.")
    agent_next_action: str = Field(default="", description="Recommended recovery action.")
    retryable: bool = Field(default=False, description="Whether retrying may succeed.")


_SEARCH_INSTRUCTION = (
    "Use normalized_documents for reasoning, summarization, comparison, and citation. "
    "Prefer complete or fallback citations. raw_response holds upstream metadata only; "
    "raw documents are omitted unless the search was run with include_raw_documents=True. "
    "Re-run with that flag only when source-specific raw fields are required."
)
_FAILURE_INSTRUCTION = (
    "Do not fabricate search results or citation identifiers. Follow the recommended recovery action."
)


@mcp_tool
class SDESearchTool(BaseTool[SDESearchToolInputSchema, SDESearchToolOutputSchema]):
    """Search NASA's Science Discovery Engine (SDE) and return citation-ready results.

    The SDE is NASA's centralized platform indexing scientific data, publications, and
    resources across sources including CMR (Earth observation), PDS (planetary science),
    SPASE (heliophysics), GCN and HEK (astronomy/solar), NAVO, OSDR (biological/physical
    sciences), code repositories, and general web documentation.

    Executes keyword, vector, or hybrid retrieval against the cross-source `generic`
    endpoint or a caller-selected source endpoint, and returns documents normalized into
    a stable contract. Citation normalization runs automatically for every document.
    Parameters are described under INPUT FIELD DESCRIPTIONS below, which the framework
    appends from the input schema.

    Every normalized document includes four citation fields, selected deterministically
    in this order: DOI, pds_lid, ivo_id, bps_osdr_id, url, missing.
    - citation_type:   doi | pds_lid | ivo_id | bps_osdr_id | url | missing
    - citation_value:  the selected DOI, identifier, or URL (empty when missing)
    - fallback_used:   false for a DOI or missing citation; true for a lower-priority identifier/URL
    - citation_status: complete (DOI) | fallback (lower-priority) | missing (none available)

    The tool never fabricates results or identifiers and never drops a valid result because
    its citation is missing. Failures are returned as a structured payload (`success=false`
    with `error_type`, `message`, `agent_next_action`, `retryable`) rather than raised, so the
    agent can recover deterministically. Summarization, query rewriting, acronym expansion,
    ranking, and internal-registry operations are the agent's responsibility, not this tool's.
    """

    input_schema = SDESearchToolInputSchema
    output_schema = SDESearchToolOutputSchema
    config_schema = SDESearchToolConfig

    # ------------------------------------------------------------------ helpers

    def _parse_document(self, doc: dict) -> NormalizedDocument:
        """Normalize one raw SDE record and attach citation fields."""

        def _s(*keys: str) -> str:
            for key in keys:
                val = doc.get(key)
                if val:
                    return str(val)
            return ""

        highlights = doc.get("highlights") or doc.get("highlight") or []
        if isinstance(highlights, dict):
            # OpenSearch-style highlight maps field -> list[str]; flatten.
            flattened: list[str] = []
            for frags in highlights.values():
                if isinstance(frags, list):
                    flattened.extend(str(f) for f in frags)
            highlights = flattened
        elif not isinstance(highlights, list):
            highlights = [str(highlights)]

        normalized = NormalizedDocument(
            id=_s("id"),
            score=float(doc.get("score") or doc.get("_score") or 0.0),
            index=_s("index", "_index", "api_source"),
            title=_s("title", "name") or "Untitled",
            url=_s("url", "readme_url"),
            division=_s("division"),
            document_type=_s("document_type", "doc_type"),
            collection_name=_s("collection_name"),
            collection_key=_s("collection_key"),
            full_text=_s("full_text"),
            data_product_desc=_s("data_product_desc"),
            relevant_content=_s("relevant_content", "description", "snippet"),
            highlights=[str(h) for h in highlights],
            persistent_id=_s("persistent_id", "doi"),
            pds_lid=_s("pds_lid"),
            ivo_id=_s("ivo_id"),
            bps_osdr_id=_s("bps_osdr_id"),
        )
        normalize_citation(normalized)
        return normalized

    def _failure(
        self,
        error_type: SDEErrorType,
        message: str,
        agent_next_action: str,
        retryable: bool,
    ) -> SDESearchToolOutputSchema:
        """Build a structured failure payload."""
        logger.debug(f"sde_search failure [{error_type}]: {message}")
        return SDESearchToolOutputSchema(
            success=False,
            error_type=error_type,
            message=message,
            agent_next_action=agent_next_action,
            retryable=retryable,
            agent_instruction=_FAILURE_INSTRUCTION,
        )

    async def _post(self, client: httpx.AsyncClient, path: str, body: dict) -> httpx.Response:
        """POST helper (kept separate so retry policy is applied by the caller)."""
        return await client.post(f"{self.config.base_url}{path}", json=body)

    async def _execute_search(
        self, client: httpx.AsyncClient, endpoint: SDESearchEndpoint, body: dict
    ) -> tuple[dict | None, SDESearchToolOutputSchema | None, int | None]:
        """Execute a search request with the retry/recovery policy.

        Returns ``(data, None, status)`` on success or ``(None, failure_payload, status)``
        on error; ``status`` is ``None`` when no response was received. HTTP 500/502/503
        and timeouts retry once; HTTP 422 vectorization failures are recoverable via
        keyword search. The caller uses ``status`` to decide whether the window is worth
        splitting (see ``_search_window``).
        """
        path = ENDPOINT_PATHS[endpoint]

        for attempt in range(2):  # initial try + at most one retry for 500/502/503
            try:
                response = await self._post(client, path, body)
            except httpx.TimeoutException:
                if attempt == 0:
                    continue
                return None, self._failure(
                    SDEErrorType.RETRYABLE_ERROR,
                    f"SDE API request timed out after {self.config.timeout}s.",
                    "Retry later.",
                    retryable=True,
                ), None
            except httpx.RequestError as e:
                return None, self._failure(
                    SDEErrorType.UPSTREAM_ERROR,
                    f"Could not connect to the SDE API: {e}",
                    "Check connectivity and retry later.",
                    retryable=True,
                ), None

            status = response.status_code
            if status < 400:
                try:
                    return response.json(), None, status
                except ValueError as e:
                    # A response cut off mid-transfer lands here; splitting the window
                    # shrinks it enough to arrive whole.
                    return None, self._failure(
                        SDEErrorType.UPSTREAM_ERROR,
                        f"SDE API returned a non-JSON response: {e}",
                        "Retry with a smaller page_size.",
                        retryable=True,
                    ), _TRUNCATED_RESPONSE

            if status in _RETRY_ONCE_STATUSES and attempt == 0:
                continue

            body_text = response.text or ""
            policy = _ERROR_POLICIES.get(status) or _unexpected_status_policy(status)
            if status == 422 and "vector" in body_text.lower():
                policy = _VECTORIZATION_POLICY
            return None, self._failure(
                policy.error_type,
                policy.message.format(status=status, body=body_text),
                policy.next_action,
                retryable=policy.retryable,
            ), status

        # Unreachable: the loop always returns.
        return None, self._failure(
            SDEErrorType.UPSTREAM_ERROR, "Search failed unexpectedly.", "Retry later.", retryable=True
        ), None

    def _build_search_body(self, params: SDESearchToolInputSchema) -> dict:
        """Assemble the SDE Search API request body from validated input."""
        body: dict = {
            "search_term": params.search_term,
            "page": params.page,
            "pageSize": params.page_size,
            "search_type": params.search_type.value,
            "include_aggregations": params.include_aggregations,
        }
        if params.filters:
            filter_payload = params.filters.to_api_payload()
            if filter_payload:
                body["filters"] = filter_payload
        return body

    @staticmethod
    def _documents_from_response(data: dict) -> list[dict]:
        """Extract the raw document list from an SDE response, tolerating shapes."""
        for key in ("documents", "results", "hits"):
            docs = data.get(key)
            if isinstance(docs, list):
                return docs
        return []

    @staticmethod
    def _prune_raw_response(data: dict) -> dict:
        """Upstream body minus keys duplicated at the top level of the output.

        The document array is fully represented by ``normalized_documents``;
        pagination/aggregations/totals are surfaced as top-level output fields.
        """
        duplicated = (
            "documents",
            "results",
            "hits",
            "pagination",
            "aggregations",
            "total_count",
            "total",
        )
        return {k: v for k, v in data.items() if k not in duplicated}

    @staticmethod
    def _sub_window_sizes(page_size: int) -> list[int]:
        """Sub-window sizes to try when a window is too large, largest first.

        Only divisors of ``page_size`` qualify. The window starts at absolute rank
        ``(page - 1) * page_size``, which is a multiple of every divisor, so sub-pages tile
        the window exactly — no document is skipped or repeated. 1 is the last resort.
        """
        sizes: list[int] = []
        size = page_size // 2
        while size >= 1:
            if page_size % size == 0:
                sizes.append(size)
            size //= 2
        if 1 not in sizes and page_size > 1:
            sizes.append(1)
        return sizes

    async def _fetch_split(
        self,
        client: httpx.AsyncClient,
        params: SDESearchToolInputSchema,
        sub_size: int,
    ) -> tuple[list[dict] | None, dict | None, SDESearchToolOutputSchema | None]:
        """Refetch the requested window as consecutive ``sub_size`` pages and concatenate.

        Returns ``(raw_documents, merged_response, None)`` or ``(None, None, failure)``.
        Rank order is preserved, so the caller's page contract is unchanged.
        """
        offset = (params.page - 1) * params.page_size
        first_page = offset // sub_size + 1

        documents: list[dict] = []
        last_data: dict | None = None
        for index in range(params.page_size // sub_size):
            body = self._build_search_body(params)
            body["page"] = first_page + index
            body["pageSize"] = sub_size
            data, failure, _ = await self._execute_search(client, params.endpoint, body)
            if failure is not None:
                return None, None, failure
            last_data = data
            page_docs = self._documents_from_response(data or {})
            documents.extend(page_docs)
            if len(page_docs) < sub_size:
                break  # ran off the end of the result set

        # Re-key the merged body so raw_response describes the whole window, not just the
        # last sub-page it happened to end on.
        merged = {
            k: v
            for k, v in (last_data or {}).items()
            if k not in ("documents", "results", "hits")
        }
        merged["documents"] = documents
        return documents, merged, None

    async def _search_window(
        self, client: httpx.AsyncClient, params: SDESearchToolInputSchema
    ) -> tuple[list[dict] | None, dict | None, SDESearchToolOutputSchema | None]:
        """Fetch the requested page, splitting it into sub-windows when it is too large."""
        body = self._build_search_body(params)
        data, failure, status = await self._execute_search(client, params.endpoint, body)
        if failure is None:
            return self._documents_from_response(data or {}), data, None
        if status not in _SPLIT_STATUSES:
            return None, None, failure

        for sub_size in self._sub_window_sizes(params.page_size):
            logger.debug(
                f"sde_search: page {params.page} (size {params.page_size}) was too large to "
                f"deliver; refetching it in pages of {sub_size}"
            )
            documents, merged, failure = await self._fetch_split(client, params, sub_size)
            if failure is None:
                return documents, merged, None
        return None, None, failure

    def _apply_response_budget(self, documents: list[NormalizedDocument]) -> int:
        """Drop ``full_text`` from the largest documents until the payload fits its budget.

        Documents themselves are never dropped — only their text, and only from the biggest
        ones — so the page contract holds and the caller can refetch an elided document on
        its own. Returns how many were elided.
        """
        limit = self.config.max_response_bytes
        if limit <= 0 or not documents:
            return 0

        payload_bytes = len(
            json.dumps([doc.model_dump(mode="json") for doc in documents], default=str)
        )
        if payload_bytes <= limit:
            return 0

        elided = 0
        for doc in sorted(documents, key=lambda d: len(d.full_text), reverse=True):
            if payload_bytes <= limit or not doc.full_text:
                break
            payload_bytes -= len(doc.full_text)
            doc.full_text_chars = len(doc.full_text)
            doc.full_text = ""
            doc.full_text_omitted = True
            elided += 1
        logger.debug(f"sde_search: dropped full_text from {elided} document(s) to fit the budget")
        return elided

    # ------------------------------------------------------------- MCP integration

    def as_function(self, mode: Literal["python", "json"] | None = None):
        """Return the MCP callable, emitting ``structuredContent`` only.

        ``mode`` is accepted for signature compatibility and ignored — this override owns
        the serialization the base class's ``mode`` would otherwise select.

        FastMCP's default conversion serializes a tool's return value into a text content
        block *and* copies it into ``structuredContent``, so every response ships the whole
        payload twice. With `full_text` included that doubling is what pushes large pages
        past the 6 MiB response limit. Returning a ``ToolResult`` short-circuits that
        conversion.

        ``content`` must be passed explicitly: ``ToolResult`` treats ``content=None`` as
        "derive it from structured_content", which reinstates the duplication. The return
        annotation stays the output schema so MCP still advertises ``outputSchema``.
        """
        inner = super().as_function(mode=None)

        async def wrapper(*args, **kwargs) -> ToolResult:
            result = await inner(*args, **kwargs)
            return ToolResult(content=[], structured_content=result.model_dump(mode="json"))

        wrapper.__name__ = inner.__name__
        wrapper.__doc__ = inner.__doc__
        wrapper.__signature__ = inner.__signature__
        wrapper.__annotations__ = dict(inner.__annotations__)
        return wrapper

    # -------------------------------------------------------------------- runner

    async def _arun(self, params: SDESearchToolInputSchema) -> SDESearchToolOutputSchema:
        """Execute the SDE search and return normalized, citation-ready results."""
        logger.debug(f"SDE search request ({params.endpoint}): {self._build_search_body(params)}")

        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            raw_documents, data, failure = await self._search_window(client, params)

        if failure is not None:
            return failure

        data = data or {}
        if not data.get("success", True) and not raw_documents:
            return self._failure(
                SDEErrorType.UPSTREAM_ERROR,
                f"SDE API returned an unsuccessful response: {str(data)[:500]}",
                "Report the upstream failure.",
                retryable=False,
            )

        if not raw_documents:
            return self._failure(
                SDEErrorType.EMPTY_RESULTS,
                "The search returned no documents.",
                "Broaden the query, remove filters, or use alternate terminology.",
                retryable=False,
            )

        documents = [self._parse_document(doc) for doc in raw_documents]
        omitted = self._apply_response_budget(documents)
        total_count = data.get("total_count", data.get("total", len(documents)))

        instruction = _SEARCH_INSTRUCTION
        if omitted:
            instruction += (
                f" full_text was dropped from {omitted} oversized document(s) to keep this "
                "response within its size limit; those documents carry full_text_omitted=true "
                "and can be refetched individually with page_size=1."
            )

        return SDESearchToolOutputSchema(
            success=True,
            endpoint_used=params.endpoint.value,
            raw_response=(
                (data if params.include_raw_documents else self._prune_raw_response(data))
                if isinstance(data, dict)
                else {}
            ),
            normalized_documents=documents,
            pagination={
                "page": params.page,
                "page_size": params.page_size,
                "total_count": total_count,
            },
            aggregations=data.get("aggregations", {}) or {},
            agent_instruction=instruction,
        )
