# SDE Search Tool (`sde_search`)

`sde_search` searches NASA's **Science Discovery Engine (SDE)** — the centralized platform indexing scientific data, publications, and resources across NASA's Science Mission Directorate — and returns documents normalized into a stable, citation-ready contract.

- **Source:** [`akd_ext/tools/sde_search.py`](../../akd_ext/tools/sde_search.py)
- **Enums / structures:** [`akd_ext/structures.py`](../../akd_ext/structures.py)
- **Tests:** [`tests/tools/test_sde_search.py`](../../tests/tools/test_sde_search.py)
- **Runnable example:** [`examples/sde_search_example.py`](../../examples/sde_search_example.py)

## What it does

1. Executes a validated request against the SDE Search API — either the cross-source `generic` endpoint or a caller-selected source endpoint.
2. Normalizes every source-specific record into a single `NormalizedDocument` shape.
3. Applies **deterministic citation normalization** to every document (DOI → `pds_lid` → `ivo_id` → `bps_osdr_id` → `url` → missing).
4. Keeps the response inside its size budget without silently dropping documents (see [Response size handling](#response-size-handling)).
5. Returns failures as a structured `success=false` payload instead of raising, so an agent can recover deterministically.

**Out of scope by design:** summarization, query rewriting, acronym expansion, ranking, and internal registry operations live in the agent/application layer, not in this tool. The tool never fabricates results or identifiers, and never drops a valid result because its citation is missing.

## Quick start

### Python

```python
import asyncio
from akd_ext.tools import SDESearchTool, SDESearchToolInputSchema

async def main():
    tool = SDESearchTool()
    result = await tool.arun(
        SDESearchToolInputSchema(search_term="mars rover", page_size=5)
    )
    if result.success:
        for doc in result.normalized_documents:
            print(doc.title, doc.citation_type, doc.citation_value)
    else:
        print(result.error_type, result.message, result.agent_next_action)

asyncio.run(main())
```

Or run the bundled example:

```bash
uv run python examples/sde_search_example.py "mars" --endpoint pds4 --page-size 3
```

### MCP server

`sde_search` is the only tool exposed on the SDE MCP surface (`akd_ext/mcp/server.py`):

```bash
# stdio transport (default)
uv run python -m akd_ext.mcp.server

# SSE transport
uv run python -m akd_ext.mcp.server --transport sse --host 127.0.0.1 --port 8000
```

Transport, host, and port can also be set via the `MCP_TRANSPORT`, `MCP_HOST`, and `MCP_PORT` environment variables.

## Configuration (`SDESearchToolConfig`)

| Field | Default | Description |
|---|---|---|
| `name` | `"sde_search"` | MCP tool name. |
| `base_url` | `$SDE_BASE_URL`, else `https://science.data.nasa.gov/science-discovery-engine` | Base URL for the SDE Search API; endpoint paths like `/api/search` are appended. |
| `timeout` | `30.0` | HTTP request timeout in seconds. |
| `max_response_bytes` | `5_000_000` | Size budget for the serialized document payload (headroom under the 6 MiB API/MCP cap). `0` disables the guard. |

## Input (`SDESearchToolInputSchema`)

| Field | Type | Default | Description |
|---|---|---|---|
| `search_term` | `str` | *(required)* | Search query. Must not be blank. |
| `endpoint` | enum | `generic` | `generic` (cross-source) or one source: `web`, `cmr`, `pds3`, `pds4`, `spase`, `gcn`, `hek`, `navo`, `osdr`, `code`. |
| `search_type` | enum | `hybrid` | Retrieval strategy: `hybrid` (vector+keyword), `keyword`, or `vector`. |
| `page` | `int` | `1` | 1-based page number. |
| `page_size` | `int` | `10` | Results per page (1–100). |
| `include_aggregations` | `bool` | `false` | Request aggregation metadata when available. |
| `include_raw_documents` | `bool` | `false` | Keep the verbatim upstream document array in `raw_response`. Off by default — `normalized_documents` already contains every document. |
| `filters` | `SDESearchFilters` | `None` | Optional result filters (see below). |

### Endpoints

| Endpoint | API path | Coverage |
|---|---|---|
| `generic` | `/api/search` | Cross-source search over all indexes |
| `web` | `/api/web/search` | General web documentation |
| `cmr` | `/api/cmr/search` | Earth observation (Common Metadata Repository) |
| `pds3` / `pds4` | `/api/pds3/search`, `/api/pds4/search` | Planetary science (Planetary Data System) |
| `spase` | `/api/spase/search` | Heliophysics |
| `gcn` | `/api/gcn/search` | Astronomy (GCN circulars) |
| `hek` | `/api/hek/search` | Solar events (Heliophysics Events Knowledgebase) |
| `navo` | `/api/navo/search` | NASA Virtual Observatory |
| `osdr` | `/api/osdr/search` | Biological & physical sciences (Open Science Data Repository) |
| `code` | `/api/code/search` | Code repositories |

### Filters (`SDESearchFilters`)

Only these four filter groups are supported. **OR** semantics apply within each list; **AND** semantics apply between groups. Empty groups are omitted from the API request.

| Field | Example values |
|---|---|
| `division` | `"Astrophysics"`, `"Planetary Science"` |
| `document_type` | `"Data"`, `"Documentation"`, `"Software and Tools"` |
| `collection_name` | Human-readable collection name |
| `collection_key` | Internal collection key |

```python
from akd_ext.tools.sde_search import SDESearchFilters

filters = SDESearchFilters(
    division=["Planetary Science"],
    document_type=["Data", "Documentation"],
)
```

## Output (`SDESearchToolOutputSchema`)

A single schema covers both success and failure so the MCP surface stays stable. `success` tells the agent which fields are populated; `agent_instruction` is always present.

### Success fields

| Field | Description |
|---|---|
| `success` | `true` |
| `agent_instruction` | Guidance on how to use the result (e.g. prefer `normalized_documents` for reasoning and citation). |
| `endpoint_used` | Endpoint value the search executed against. |
| `normalized_documents` | List of `NormalizedDocument` (below). |
| `pagination` | `{page, page_size, total_count}`. |
| `aggregations` | Aggregation metadata when requested/available. |
| `raw_response` | Upstream response metadata. The document array and keys duplicated at the top level (pagination, aggregations, totals) are pruned unless `include_raw_documents=true`. |

### Failure fields

| Field | Description |
|---|---|
| `success` | `false` |
| `error_type` | `validation_error` \| `upstream_error` \| `empty_results` \| `retryable_error` |
| `message` | Human-readable error message (includes upstream HTTP status/body where relevant). |
| `agent_next_action` | Recommended recovery action. |
| `retryable` | Whether retrying may succeed. |

### `NormalizedDocument`

Fields common across index types, plus the four citation fields:

| Field | Description |
|---|---|
| `id` | Normalized document identifier. |
| `score` | Relevance score from the search engine. |
| `index` | Source index the record came from. |
| `title` | Document or dataset title (`"Untitled"` when the source has none). |
| `url` | Direct URL to access the document. |
| `division`, `document_type`, `collection_name`, `collection_key` | SMD/collection metadata. |
| `full_text` | Full text or abstract, when available. **Never truncated** — it is dropped whole only under the size budget (see below). |
| `full_text_omitted` | `true` when `full_text` was dropped to fit the size budget. |
| `full_text_chars` | Length of the dropped `full_text`; `0` unless omitted. |
| `data_product_desc` | Data product description, when available. |
| `relevant_content` | Most relevant snippet for the query. |
| `highlights` | Highlighted matching fragments. |
| `persistent_id`, `pds_lid`, `ivo_id`, `bps_osdr_id` | Source persistent identifiers. |
| `citation_type`, `citation_value`, `fallback_used`, `citation_status` | Citation fields (below). |

## Citation normalization

Every returned document gets four citation fields, selected deterministically by a fixed identifier hierarchy — first valid, non-empty value wins:

1. **DOI** — extracted from `persistent_id` (recognizes bare `10.xxxx/...`, `https://doi.org/...`, and `doi:` forms; when a field packs several DOIs, the first is used)
2. `pds_lid`
3. `ivo_id`
4. `bps_osdr_id`
5. external `url`
6. missing

| Outcome | `citation_type` | `citation_status` | `fallback_used` |
|---|---|---|---|
| DOI found | `doi` | `complete` | `false` |
| Lower-priority identifier or URL | `pds_lid` / `ivo_id` / `bps_osdr_id` / `url` | `fallback` | `true` |
| Nothing available | `missing` | `missing` | `false` |

There is no separate citation-lookup operation: the SDE Search API has no lookup-by-identifier endpoint, so citations are produced only from search results, where every field the hierarchy needs is already present. Identifiers are never fabricated or inferred.

## Error handling & retry policy

Failures are returned as structured payloads, never raised past the MCP boundary.

| Condition | `error_type` | `retryable` | Behavior / recommended action |
|---|---|---|---|
| HTTP 400 | `validation_error` | no | Correct the request; do not retry unchanged. |
| HTTP 422 (schema) | `validation_error` | no | Correct the request schema. |
| HTTP 422 mentioning vectorization | `retryable_error` | yes | Retry with `search_type="keyword"`. |
| HTTP 500 | `upstream_error` | no | Retried once automatically, then reported. |
| HTTP 502 | `retryable_error` | yes | Retried once, then the window is split (see below). |
| HTTP 503 | `retryable_error` | yes | Retried once automatically; retry later. |
| Timeout | `retryable_error` | yes | Retried once automatically; retry later. |
| Connection error | `upstream_error` | yes | Check connectivity and retry later. |
| Truncated/non-JSON 200 body | `upstream_error` | yes | Treated as oversized; the window is split. |
| Zero documents | `empty_results` | no | Broaden the query, remove filters, or use alternate terminology. |
| Other 4xx/5xx | `upstream_error` | 5xx only | Report the upstream failure. |

## Response size handling

Both the SDE API and the MCP host cap a response at **6 MiB**, and `full_text` is unbounded per document, so a single page can exceed the limit. The tool keeps responses within bounds in three layers:

1. **`structuredContent` only.** FastMCP's default conversion ships the payload twice (text block + `structuredContent`); the tool's `as_function` override emits `structuredContent` only, halving response size.
2. **Window splitting.** If a page is too large to deliver (HTTP 502 or a truncated body), the same window is refetched as smaller, exactly aligned sub-pages (divisors of `page_size`, largest first, down to 1) and concatenated. Rank order is preserved, so the page contract is unchanged.
3. **`full_text` elision.** If the serialized payload still exceeds `max_response_bytes` (default 5 MB), `full_text` is dropped whole from the largest documents until it fits. Those documents carry `full_text_omitted=true` and `full_text_chars`, and can be refetched individually with `page_size=1`. Documents themselves are **never** dropped, and `full_text` is never partially truncated.

## Testing

```bash
uv run pytest tests/tools/test_sde_search.py
```
