# test_run.py
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# tvoj public interface (wrapper)
from pipeline.search_everywhere import search_everywhere


def print_section(title: str):
    print("\n" + "=" * 60)
    print(title.upper())
    print("=" * 60)


def print_docs(label: str, docs):
    print_section(label)
    if not docs:
        print("  (no results)\n")
        return
    for d in docs[:5]:  # prikažemo do 5 rezultata po kategoriji
        print(f"- {d.metadata.get('title') or d.metadata.get('source') or 'doc'}")
        print(f"  Source: {d.metadata.get('url', 'local')}")
        preview = (d.page_content[:200] + "...").replace("\n", " ")
        print(f"  Text: {preview}")
        print()


def main():
    print_section("STARTING INTEGRATION TEST")

    # učitavamo .env
    load_dotenv()

    query = "machine learning"
    print(f"Running search_everywhere(query='{query}')\n")

    # pozivamo tvoj RAG search pipeline
    results = search_everywhere(query)

    # results je dict: { "wikipedia": [...], "stackoverflow": [...], "openalex": [...], "pdf": [...] }
    for key, docs in results.items():
        print_docs(key, docs)

    print_section("DONE")
    print("Integration test finished successfully.\n")


if __name__ == "__main__":
    main()
