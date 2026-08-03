"""Tests for the SDE Search Tool (`sde_search` MCP contract)."""

import json

import httpx
import jsonschema
import pytest
from fastmcp.tools.tool import ToolResult

from akd_ext.structures import (
    SDECitationStatus,
    SDECitationType,
    SDEErrorType,
    SDESearchEndpoint,
)
from akd_ext.tools import SDESearchTool, SDESearchToolInputSchema
from akd_ext.tools.sde_search import (
    NormalizedDocument,
    SDESearchToolOutputSchema,
    _extract_doi,
    normalize_citation,
)

# ---------------------------------------------------------------------------
# Unit tests — deterministic, no network
# ---------------------------------------------------------------------------


class TestDoiExtraction:
    def test_bare_doi(self):
        assert _extract_doi("10.1007/s11214-021-00816-9") == "10.1007/s11214-021-00816-9"

    def test_doi_url_prefix(self):
        assert _extract_doi("https://doi.org/10.5067/ABC-123") == "10.5067/ABC-123"

    def test_doi_scheme_prefix(self):
        assert _extract_doi("doi:10.5067/ABC-123") == "10.5067/ABC-123"

    def test_first_of_multiple_dois(self):
        # SDE persistent_id fields sometimes pack several DOIs separated by `;`.
        value = "10.1007/s11214-021-00816-9;10.1007/s11214-020-00762-y"
        assert _extract_doi(value) == "10.1007/s11214-021-00816-9"

    def test_non_doi_persistent_id_rejected(self):
        assert _extract_doi("urn:nasa:pds:mars2020_meda:document") is None

    def test_none_and_empty(self):
        assert _extract_doi(None) is None
        assert _extract_doi("") is None


class TestCitationHierarchy:
    def test_doi_is_complete(self):
        doc = NormalizedDocument(persistent_id="https://doi.org/10.1234/abcd")
        normalize_citation(doc)
        assert doc.citation_type == SDECitationType.DOI
        assert doc.citation_status == SDECitationStatus.COMPLETE
        assert doc.fallback_used is False
        assert doc.citation_value == "10.1234/abcd"

    def test_pds_lid_fallback(self):
        doc = NormalizedDocument(pds_lid="urn:nasa:pds:foo")
        normalize_citation(doc)
        assert doc.citation_type == SDECitationType.PDS_LID
        assert doc.citation_status == SDECitationStatus.FALLBACK
        assert doc.fallback_used is True

    def test_hierarchy_prefers_doi_over_lower(self):
        doc = NormalizedDocument(
            persistent_id="10.1234/abc",
            pds_lid="urn:nasa:pds:foo",
            ivo_id="ivo://x",
        )
        normalize_citation(doc)
        assert doc.citation_type == SDECitationType.DOI

    def test_ivo_then_bps_then_url_order(self):
        doc = NormalizedDocument(ivo_id="ivo://x", bps_osdr_id="OSD-1", url="http://x")
        normalize_citation(doc)
        assert doc.citation_type == SDECitationType.IVO_ID

        doc = NormalizedDocument(bps_osdr_id="OSD-1", url="http://x")
        normalize_citation(doc)
        assert doc.citation_type == SDECitationType.BPS_OSDR_ID

        doc = NormalizedDocument(url="http://x")
        normalize_citation(doc)
        assert doc.citation_type == SDECitationType.URL
        assert doc.fallback_used is True

    def test_missing(self):
        doc = NormalizedDocument()
        normalize_citation(doc)
        assert doc.citation_type == SDECitationType.MISSING
        assert doc.citation_status == SDECitationStatus.MISSING
        assert doc.citation_value == ""
        assert doc.fallback_used is False


class TestInputValidation:
    def test_search_term_required(self):
        with pytest.raises(Exception):
            SDESearchToolInputSchema()
        with pytest.raises(Exception):
            SDESearchToolInputSchema(search_term="")
        with pytest.raises(Exception):
            SDESearchToolInputSchema(search_term="   ")

    def test_search_defaults(self):
        p = SDESearchToolInputSchema(search_term="mars rover")
        assert p.endpoint == SDESearchEndpoint.GENERIC
        assert p.page == 1
        assert p.page_size == 10

    def test_page_size_bounds(self):
        with pytest.raises(Exception):
            SDESearchToolInputSchema(search_term="x", page_size=0)
        with pytest.raises(Exception):
            SDESearchToolInputSchema(search_term="x", page_size=101)
        with pytest.raises(Exception):
            SDESearchToolInputSchema(search_term="x", page=0)


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text

    def json(self):
        return self._payload


def _patch_post(monkeypatch, responses: list):
    """Patch SDESearchTool._post to yield queued responses (or exceptions)."""
    queue = list(responses)

    async def fake_post(self, client, path, body):
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(SDESearchTool, "_post", fake_post)


class TestSearchMocked:
    async def test_successful_search_normalizes_and_cites(self, monkeypatch):
        payload = {
            "success": True,
            "total_count": 2,
            "documents": [
                {
                    "id": "a1",
                    "_score": 9.5,
                    "_index": "sde-pds4",
                    "title": "MEDA Data",
                    "persistent_id": "10.17189/1522643",
                    "pds_lid": "urn:nasa:pds:mars2020_meda:data",
                },
                {
                    "id": "b2",
                    "_index": "sde-web",
                    "title": "Web Page",
                    "url": "https://example.org/page",
                },
            ],
        }
        _patch_post(monkeypatch, [_FakeResponse(200, payload)])
        tool = SDESearchTool()
        out = await tool.arun(SDESearchToolInputSchema(search_term="mars", endpoint="pds4"))

        assert out.success is True
        assert out.endpoint_used == "pds4"
        assert out.pagination["total_count"] == 2
        assert len(out.normalized_documents) == 2
        # Citation fields normalized onto every document.
        assert out.normalized_documents[0].citation_type == SDECitationType.DOI
        assert out.normalized_documents[0].citation_status == SDECitationStatus.COMPLETE
        assert out.normalized_documents[0].fallback_used is False
        assert out.normalized_documents[1].citation_type == SDECitationType.URL
        assert out.normalized_documents[1].citation_status == SDECitationStatus.FALLBACK

    async def test_empty_results(self, monkeypatch):
        _patch_post(monkeypatch, [_FakeResponse(200, {"success": True, "documents": []})])
        tool = SDESearchTool()
        out = await tool.arun(SDESearchToolInputSchema(search_term="zzznomatch"))
        assert out.success is False
        assert out.error_type == SDEErrorType.EMPTY_RESULTS
        assert out.retryable is False

    async def test_http_400_validation_error_not_retryable(self, monkeypatch):
        _patch_post(monkeypatch, [_FakeResponse(400, text="bad request")])
        tool = SDESearchTool()
        out = await tool.arun(SDESearchToolInputSchema(search_term="mars"))
        assert out.success is False
        assert out.error_type == SDEErrorType.VALIDATION_ERROR
        assert out.retryable is False

    async def test_http_422_vectorization_is_retryable(self, monkeypatch):
        _patch_post(monkeypatch, [_FakeResponse(422, text="vector embedding failed")])
        tool = SDESearchTool()
        out = await tool.arun(SDESearchToolInputSchema(search_term="mars"))
        assert out.error_type == SDEErrorType.RETRYABLE_ERROR
        assert out.retryable is True
        assert "keyword" in out.agent_next_action.lower()

    async def test_http_422_schema_is_validation_error(self, monkeypatch):
        _patch_post(monkeypatch, [_FakeResponse(422, text="field required")])
        tool = SDESearchTool()
        out = await tool.arun(SDESearchToolInputSchema(search_term="mars"))
        assert out.error_type == SDEErrorType.VALIDATION_ERROR
        assert out.retryable is False

    async def test_http_500_retries_once_then_fails(self, monkeypatch):
        _patch_post(monkeypatch, [_FakeResponse(500, text="boom"), _FakeResponse(500, text="boom")])
        tool = SDESearchTool()
        out = await tool.arun(SDESearchToolInputSchema(search_term="mars"))
        assert out.error_type == SDEErrorType.UPSTREAM_ERROR

    async def test_http_500_retry_then_success(self, monkeypatch):
        ok = {"success": True, "documents": [{"id": "a", "title": "T", "url": "http://x"}]}
        _patch_post(monkeypatch, [_FakeResponse(500, text="boom"), _FakeResponse(200, ok)])
        tool = SDESearchTool()
        out = await tool.arun(SDESearchToolInputSchema(search_term="mars"))
        assert out.success is True
        assert len(out.normalized_documents) == 1

    async def test_http_503_retryable(self, monkeypatch):
        _patch_post(monkeypatch, [_FakeResponse(503), _FakeResponse(503)])
        tool = SDESearchTool()
        out = await tool.arun(SDESearchToolInputSchema(search_term="mars"))
        assert out.error_type == SDEErrorType.RETRYABLE_ERROR
        assert out.retryable is True

    async def test_timeout_retries_then_retryable(self, monkeypatch):
        _patch_post(
            monkeypatch,
            [httpx.TimeoutException("t1"), httpx.TimeoutException("t2")],
        )
        tool = SDESearchTool()
        out = await tool.arun(SDESearchToolInputSchema(search_term="mars"))
        assert out.error_type == SDEErrorType.RETRYABLE_ERROR


class _TruncatedResponse(_FakeResponse):
    """HTTP 200 whose body was cut off in transit, so it does not parse."""

    def json(self):
        raise ValueError("Unterminated string starting at: line 1 column 55 (char 54)")


def _docs(*ids, chars: int = 0) -> dict:
    return {
        "success": True,
        "total_count": 40,
        "documents": [{"id": i, "title": f"T{i}", "url": f"https://x/{i}", "full_text": "z" * chars} for i in ids],
    }


def _patch_post_capturing(monkeypatch, responses: list) -> list[dict]:
    """Like `_patch_post`, but records each request body so paging can be asserted."""
    queue, sent = list(responses), []

    async def fake_post(self, client, path, body):
        sent.append(dict(body))
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(SDESearchTool, "_post", fake_post)
    return sent


class TestOversizeWindowSplitting:
    """A window too large to deliver is refetched in smaller aligned pages.

    Both the SDE API and the MCP host cap a response at 6 MiB. The API signals it with a
    502, and a partially delivered body fails to parse; either way the same documents come
    back when the window is fetched in pieces.
    """

    def test_sub_window_sizes_are_divisors_largest_first(self):
        assert SDESearchTool._sub_window_sizes(10) == [5, 2, 1]
        assert SDESearchTool._sub_window_sizes(8) == [4, 2, 1]
        # A prime page_size can only fall back to one document at a time.
        assert SDESearchTool._sub_window_sizes(7) == [1]
        # Nothing smaller than a single document to try.
        assert SDESearchTool._sub_window_sizes(1) == []

    async def test_502_splits_window_and_preserves_rank_order(self, monkeypatch):
        sent = _patch_post_capturing(
            monkeypatch,
            [
                _FakeResponse(502, text='{"message": "Internal server error"}'),
                _FakeResponse(502, text='{"message": "Internal server error"}'),
                _FakeResponse(200, _docs("a", "b", "c", "d", "e")),
                _FakeResponse(200, _docs("f", "g", "h", "i", "j")),
            ],
        )
        tool = SDESearchTool()
        out = await tool.arun(SDESearchToolInputSchema(search_term="mars"))

        assert out.success is True
        assert [d.id for d in out.normalized_documents] == list("abcdefghij")
        # Full window tried twice (502 retries once), then two aligned half-pages.
        assert [(b["page"], b["pageSize"]) for b in sent] == [(1, 10), (1, 10), (1, 5), (2, 5)]

    async def test_split_pages_align_to_the_requested_offset(self, monkeypatch):
        """Page 2 of size 10 is ranks 11-20, i.e. pages 3 and 4 of size 5."""
        sent = _patch_post_capturing(
            monkeypatch,
            [
                _FakeResponse(502, text="err"),
                _FakeResponse(502, text="err"),
                _FakeResponse(200, _docs("k", "l", "m", "n", "o")),
                _FakeResponse(200, _docs("p", "q", "r", "s", "t")),
            ],
        )
        tool = SDESearchTool()
        out = await tool.arun(SDESearchToolInputSchema(search_term="mars", page=2))

        assert out.success is True
        assert [(b["page"], b["pageSize"]) for b in sent][2:] == [(3, 5), (4, 5)]
        assert out.pagination["page"] == 2

    async def test_truncated_body_also_splits(self, monkeypatch):
        _patch_post_capturing(
            monkeypatch,
            [
                _TruncatedResponse(200),
                _FakeResponse(200, _docs("a", "b", "c", "d", "e")),
                _FakeResponse(200, _docs("f", "g", "h", "i", "j")),
            ],
        )
        tool = SDESearchTool()
        out = await tool.arun(SDESearchToolInputSchema(search_term="mars"))
        assert out.success is True
        assert len(out.normalized_documents) == 10

    async def test_short_final_page_stops_the_split_early(self, monkeypatch):
        sent = _patch_post_capturing(
            monkeypatch,
            [
                _FakeResponse(502, text="err"),
                _FakeResponse(502, text="err"),
                _FakeResponse(200, _docs("a", "b")),  # fewer than sub_size -> end of results
            ],
        )
        tool = SDESearchTool()
        out = await tool.arun(SDESearchToolInputSchema(search_term="mars"))
        assert [d.id for d in out.normalized_documents] == ["a", "b"]
        assert len(sent) == 3  # did not request the second half

    async def test_non_splittable_status_is_returned_as_is(self, monkeypatch):
        """A 400 is a client error — splitting it would just repeat the mistake."""
        sent = _patch_post_capturing(monkeypatch, [_FakeResponse(400, text="bad query")])
        tool = SDESearchTool()
        out = await tool.arun(SDESearchToolInputSchema(search_term="mars"))
        assert out.success is False
        assert out.error_type == SDEErrorType.VALIDATION_ERROR
        assert len(sent) == 1


class TestResponseBudget:
    async def test_oversized_payload_elides_largest_full_text_only(self, monkeypatch):
        _patch_post(
            monkeypatch,
            [
                _FakeResponse(
                    200,
                    {
                        "success": True,
                        "total_count": 3,
                        "documents": [
                            {"id": "small", "title": "S", "full_text": "s" * 50},
                            {"id": "huge", "title": "H", "full_text": "h" * 8000},
                            {"id": "big", "title": "B", "full_text": "b" * 4000},
                        ],
                    },
                )
            ],
        )
        tool = SDESearchTool(max_response_bytes=5000)
        out = await tool.arun(SDESearchToolInputSchema(search_term="mars"))

        by_id = {d.id: d for d in out.normalized_documents}
        assert set(by_id) == {"small", "huge", "big"}, "no document may be dropped"
        assert by_id["huge"].full_text_omitted is True
        assert by_id["huge"].full_text == ""
        assert by_id["huge"].full_text_chars == 8000
        # Only as much is shed as the budget requires — the smallest keeps its text.
        assert by_id["small"].full_text_omitted is False
        assert by_id["small"].full_text == "s" * 50
        assert "full_text was dropped" in out.agent_instruction

    async def test_payload_within_budget_is_untouched(self, monkeypatch):
        _patch_post(monkeypatch, [_FakeResponse(200, _docs("a", "b", chars=100))])
        tool = SDESearchTool(max_response_bytes=5_000_000)
        out = await tool.arun(SDESearchToolInputSchema(search_term="mars"))
        assert all(not d.full_text_omitted for d in out.normalized_documents)
        assert all(d.full_text for d in out.normalized_documents)
        assert "full_text was dropped" not in out.agent_instruction

    async def test_budget_of_zero_disables_the_guard(self, monkeypatch):
        _patch_post(monkeypatch, [_FakeResponse(200, _docs("a", chars=9000))])
        tool = SDESearchTool(max_response_bytes=0)
        out = await tool.arun(SDESearchToolInputSchema(search_term="mars"))
        assert out.normalized_documents[0].full_text == "z" * 9000


class TestMcpContract:
    def test_tool_name_is_sde_search(self):
        assert SDESearchTool().name == "sde_search"

    async def test_as_function_emits_structured_content_only(self, monkeypatch):
        """MCP otherwise ships the whole payload twice, doubling it against a 6 MiB cap."""
        _patch_post(monkeypatch, [_FakeResponse(200, _docs("a", "b", chars=1000))])
        result = await SDESearchTool().as_function(mode="python")(search_term="mars")

        assert isinstance(result, ToolResult)
        assert result.content == [], "content must stay empty; it would duplicate the payload"
        assert result.structured_content["success"] is True
        assert len(result.structured_content["normalized_documents"]) == 2
        # to_mcp_result yields (content, structured_content) rather than content alone.
        assert isinstance(result.to_mcp_result(), tuple)

    async def test_full_text_survives_the_dedup(self, monkeypatch):
        _patch_post(monkeypatch, [_FakeResponse(200, _docs("a", chars=1000))])
        result = await SDESearchTool().as_function(mode="python")(search_term="mars")
        doc = result.structured_content["normalized_documents"][0]
        assert doc["full_text"] == "z" * 1000

    def test_omitting_content_would_reinstate_the_duplication(self):
        """Guards the ToolResult behaviour the dedup depends on.

        `ToolResult` treats `content=None` as "derive it from structured_content", so
        dropping the explicit `content=[]` silently doubles the payload again.
        """
        payload = {"success": True, "normalized_documents": [{"id": "a"}]}
        assert ToolResult(content=[], structured_content=payload).content == []
        assert ToolResult(structured_content=payload).content != []

    async def test_payload_is_json_native_and_matches_output_schema(self, monkeypatch):
        """Bypassing FastMCP's conversion means we serialize the payload ourselves."""
        _patch_post(monkeypatch, [_FakeResponse(200, _docs("a"))])
        result = await SDESearchTool().as_function(mode="python")(search_term="mars")
        payload = result.structured_content

        # Enums must land as strings, not enum objects.
        assert payload["normalized_documents"][0]["citation_type"] == "url"
        json.dumps(payload)  # must not raise
        jsonschema.validate(
            instance=payload,
            schema=SDESearchToolOutputSchema.model_json_schema(mode="serialization"),
        )

    def test_output_schema_is_still_advertised(self):
        """The return annotation stays the output schema so MCP still publishes it."""
        fn = SDESearchTool().as_function(mode="python")
        assert fn.__annotations__["return"] is SDESearchToolOutputSchema
        assert fn.__name__ == "sde_search"


# ---------------------------------------------------------------------------
# Integration tests — hit the live SDE Search API
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_live_search_basic():
    tool = SDESearchTool()
    out = await tool.arun(SDESearchToolInputSchema(search_term="climate change", page_size=5))
    assert out.success is True
    assert 0 < len(out.normalized_documents) <= 5
    for doc in out.normalized_documents:
        assert doc.title
        assert doc.citation_type in set(SDECitationType)
        assert doc.citation_status in set(SDECitationStatus)


@pytest.mark.integration
async def test_live_search_source_endpoint():
    tool = SDESearchTool()
    out = await tool.arun(SDESearchToolInputSchema(search_term="mars", endpoint="pds4", page_size=5))
    assert out.success is True
    assert out.endpoint_used == "pds4"


@pytest.mark.integration
async def test_live_search_division_filter():
    tool = SDESearchTool()
    out = await tool.arun(
        SDESearchToolInputSchema(
            search_term="climate change",
            page_size=5,
            filters={"division": ["Earth Science"]},
        )
    )
    assert out.success is True
    for doc in out.normalized_documents:
        assert doc.division == "Earth Science"
