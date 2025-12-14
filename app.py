from pipeline.search_everywhere import search_everywhere

res = search_everywhere("What is Python programming?", limits={"gcs": 3})
print("Keys:", res.keys())
print("GCS docs:", len(res.get("gcs", [])))
for d in res.get("gcs", [])[:2]:
    print("TITLE:", d.metadata.get("title"))
    print("TEXT:", d.page_content[:200])
    print("---")
