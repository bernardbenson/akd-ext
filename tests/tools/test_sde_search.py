"""Tests for the SDE Search Tool (`sde_search` MCP contract)."""

import httpx
import pytest

from akd_ext.structures import (
    SDECitationStatus,
    SDECitationType,
    SDEErrorType,
    SDESearchEndpoint,
)
from akd_ext.tools import SDESearchTool, SDESearchToolInputSchema
from akd_ext.tools.sde_search import (
    NormalizedDocument,
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


class TestMcpContract:
    def test_tool_name_is_sde_search(self):
        assert SDESearchTool().name == "sde_search"


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
