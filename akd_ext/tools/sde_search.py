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

Boundaries: summarization, query rewriting, acronym expansion, ranking, and internal
registry operations live in the agent/application layer, not in this tool.
"""

import os
import re

import httpx
from loguru import logger
from pydantic import BaseModel, Field, model_validator

from akd._base import InputSchema, OutputSchema
from akd.tools import BaseTool, BaseToolConfig

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
        default="", description="Full text or abstract, when available. Never truncated."
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
            "SDE_BASE_URL", "https://dyejsbdumgpqz.cloudfront.net"
        ),
        description="Base URL for the SDE Search API (endpoint paths like /api/search are appended).",
    )
    timeout: float = Field(default=30.0, description="HTTP request timeout in seconds.")


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
    - search_term: the query (required, non-blank)
    - endpoint: generic | web | cmr | pds3 | pds4 | spase | gcn | hek | navo | osdr | code
    - search_type: hybrid (default) | keyword | vector
    - page / page_size: pagination (page >= 1, page_size 1-100)
    - include_aggregations: request aggregation metadata when available
    - include_raw_documents: include the verbatim upstream document array in raw_response
      (off by default; normalized_documents already contains every document, and
      raw_response otherwise carries upstream metadata only)
    - filters: division, document_type, collection_name, collection_key
               (OR within a list, AND across groups)

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
            id=str(doc.get("_id") or ""),
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
    ) -> tuple[dict | None, SDESearchToolOutputSchema | None]:
        """Execute a search request with the retry/recovery policy.

        Returns ``(data, None)`` on success or ``(None, failure_payload)`` on error.
        HTTP 500/503 and timeouts retry once; HTTP 422 vectorization failures are
        recoverable via keyword search.
        """
        path = ENDPOINT_PATHS[endpoint]

        for attempt in range(2):  # initial try + at most one retry for 500/503
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
                )
            except httpx.RequestError as e:
                return None, self._failure(
                    SDEErrorType.UPSTREAM_ERROR,
                    f"Could not connect to the SDE API: {e}",
                    "Check connectivity and retry later.",
                    retryable=True,
                )

            status = response.status_code
            if status < 400:
                try:
                    return response.json(), None
                except ValueError as e:
                    return None, self._failure(
                        SDEErrorType.UPSTREAM_ERROR,
                        f"SDE API returned a non-JSON response: {e}",
                        "Report the upstream failure.",
                        retryable=False,
                    )

            body_text = response.text or ""
            if status == 400:
                return None, self._failure(
                    SDEErrorType.VALIDATION_ERROR,
                    f"SDE API rejected the request (HTTP 400): {body_text}",
                    "Correct the request; do not retry unchanged.",
                    retryable=False,
                )
            if status == 422:
                if "vector" in body_text.lower():
                    return None, self._failure(
                        SDEErrorType.RETRYABLE_ERROR,
                        f"Vectorization failed (HTTP 422): {body_text}",
                        "Retry using keyword search (search_type='keyword').",
                        retryable=True,
                    )
                return None, self._failure(
                    SDEErrorType.VALIDATION_ERROR,
                    f"SDE API schema validation failed (HTTP 422): {body_text}",
                    "Correct the request schema.",
                    retryable=False,
                )
            if status == 500:
                if attempt == 0:
                    continue
                return None, self._failure(
                    SDEErrorType.UPSTREAM_ERROR,
                    f"SDE API internal error (HTTP 500): {body_text}",
                    "Report the upstream failure.",
                    retryable=False,
                )
            if status == 503:
                if attempt == 0:
                    continue
                return None, self._failure(
                    SDEErrorType.RETRYABLE_ERROR,
                    f"SDE API unavailable (HTTP 503): {body_text}",
                    "Report service unavailability; retry later.",
                    retryable=True,
                )
            # Other 4xx/5xx.
            return None, self._failure(
                SDEErrorType.UPSTREAM_ERROR,
                f"SDE API returned HTTP {status}: {body_text}",
                "Report the upstream failure.",
                retryable=status >= 500,
            )

        # Unreachable: the loop always returns.
        return None, self._failure(
            SDEErrorType.UPSTREAM_ERROR, "Search failed unexpectedly.", "Retry later.", retryable=True
        )

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

    # -------------------------------------------------------------------- runner

    async def _arun(self, params: SDESearchToolInputSchema) -> SDESearchToolOutputSchema:
        """Execute the SDE search and return normalized, citation-ready results."""
        body = self._build_search_body(params)
        logger.debug(f"SDE search request ({params.endpoint}): {body}")

        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            data, failure = await self._execute_search(client, params.endpoint, body)

        if failure is not None:
            return failure

        assert data is not None
        if not data.get("success", True) and not self._documents_from_response(data):
            return self._failure(
                SDEErrorType.UPSTREAM_ERROR,
                f"SDE API returned an unsuccessful response: {str(data)[:500]}",
                "Report the upstream failure.",
                retryable=False,
            )

        raw_documents = self._documents_from_response(data)
        if not raw_documents:
            return self._failure(
                SDEErrorType.EMPTY_RESULTS,
                "The search returned no documents.",
                "Broaden the query, remove filters, or use alternate terminology.",
                retryable=False,
            )

        documents = [self._parse_document(doc) for doc in raw_documents]
        total_count = data.get("total_count", data.get("total", len(documents)))

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
            agent_instruction=_SEARCH_INSTRUCTION,
        )
