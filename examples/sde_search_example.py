"""Example: call the ``sde_search`` tool and display its input and output.

Runs a live search against the SDE Search API and prints the request (input
schema) and the response (output schema), including the four citation fields that
are normalized onto every returned document.

Run with (defaults shown):
    uv run python examples/sde_search_example.py
    uv run python examples/sde_search_example.py "mars rover"
    uv run python examples/sde_search_example.py "mars" --endpoint pds4 --page-size 3
"""

from __future__ import annotations

import argparse
import asyncio
import json

from akd_ext.structures import SDESearchEndpoint
from akd_ext.tools import SDESearchTool, SDESearchToolInputSchema


def _hr(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _dump(model) -> str:
    """Pretty-print a pydantic model as JSON (enums -> their string values)."""
    return json.dumps(model.model_dump(mode="json"), indent=2, default=str)


async def run_search(tool: SDESearchTool, request: SDESearchToolInputSchema) -> None:
    """Execute a search and print the input and output."""
    _hr("SEARCH — INPUT")
    print(_dump(request))

    result = await tool.arun(request)

    _hr("SEARCH — OUTPUT (summary)")
    print(f"success:        {result.success}")
    print(f"endpoint_used:  {result.endpoint_used}")
    print(f"pagination:     {result.pagination}")
    print(f"documents:      {len(result.normalized_documents)}")

    if not result.success:
        print(f"error_type:     {result.error_type}")
        print(f"message:        {result.message}")
        print(f"next_action:    {result.agent_next_action}")
        return

    _hr("SEARCH — NORMALIZED DOCUMENTS")
    for i, doc in enumerate(result.normalized_documents, 1):
        print(f"\n[{i}] {doc.title}")
        print(f"    url:            {doc.url or '-'}")
        print(f"    division:       {doc.division or '-'}")
        print(f"    document_type:  {doc.document_type or '-'}")
        print(f"    score:          {doc.score}")
        # The four citation fields, normalized onto every document.
        print(
            f"    citation:       type={doc.citation_type.value} "
            f"status={doc.citation_status.value} "
            f"fallback_used={doc.fallback_used}"
        )
        print(f"    citation_value: {doc.citation_value or '(none)'}")

    print("\nagent_instruction:")
    print(f"  {result.agent_instruction}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Call the sde_search tool and display its I/O.")
    parser.add_argument(
        "search_term",
        nargs="?",
        default="Broeggerhalvoya",
        help="Search query (default: 'Broeggerhalvoya').",
    )
    parser.add_argument(
        "--endpoint",
        choices=[e.value for e in SDESearchEndpoint],
        default=SDESearchEndpoint.GENERIC.value,
        help="Search endpoint (default: generic).",
    )
    parser.add_argument(
        "--search-type",
        choices=["hybrid", "keyword", "vector"],
        default="hybrid",
        help="Retrieval strategy (default: hybrid).",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=5,
        help="Results per page, 1-100 (default: 5).",
    )
    return parser.parse_args()


async def main() -> None:
    args = _parse_args()
    tool = SDESearchTool()
    print(f"MCP tool name: {tool.name}")
    print(f"Base URL:      {tool.config.base_url}")

    request = SDESearchToolInputSchema(
        search_term=args.search_term,
        endpoint=args.endpoint,
        search_type=args.search_type,
        page=1,
        page_size=args.page_size,
        include_aggregations=False,
    )
    await run_search(tool, request)


if __name__ == "__main__":
    asyncio.run(main())
