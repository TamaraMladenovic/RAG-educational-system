from __future__ import annotations

from typing import Dict, List, Optional

#Ukoliko dođe do nepoklapanja verzija
try:   
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document 

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter


from pipeline.common import clean_text


# Konfiguracija po tipu dokumenta
CHUNK_CONFIG: Dict[str, Dict[str, int]] = {
    "pdf": {"chunk_size": 900, "chunk_overlap": 150},
    "wikipedia": {"chunk_size": 1000, "chunk_overlap": 200},
    "stackoverflow": {"chunk_size": 700, "chunk_overlap": 120},
    "openalex": {"chunk_size": 1200, "chunk_overlap": 200},
    "generic": {"chunk_size": 800, "chunk_overlap": 200},
}

DEFAULT_SEPARATORS = [
    "\n\n",
    "\n",
    ". ",
    "? ",
    "! ",
    " ",
    "",
]


def chunk_clean(text: str) -> str:    #Priprema tekst za chunking - uklanjanje praznih linija

    cleaned = clean_text(text)

    lines = [line.strip() for line in cleaned.split("\n")]

    filtered = []
    for line in lines:
        if line == "" and (not filtered or filtered[-1] == ""):
            continue
        filtered.append(line)

    return "\n".join(filtered)


def detect_doc_type(doc: Document) -> str:  #Prepoznavanje tipa dokumenta

    meta = doc.metadata or {}
    source = str(
        meta.get("source")
        or meta.get("provider")
        or meta.get("source_type")
        or ""
    ).lower()

    file_type = str(meta.get("file_type") or meta.get("extension") or "").lower()

    if "wikipedia" in source:
        return "wikipedia"
    if "stack overflow" in source or "stackoverflow" in source:
        return "stackoverflow"
    if "openalex" in source:
        return "openalex"
    if "pdf" in file_type or file_type.endswith(".pdf"):
        return "pdf"

    return "generic"


def build_text_splitter_for_type(
    doc_type: str,
    override_chunk_size: Optional[int] = None,
    override_chunk_overlap: Optional[int] = None,
) -> RecursiveCharacterTextSplitter:       #Rekurzivni text splitter na osnovu tipa dokumenta

    cfg = CHUNK_CONFIG.get(doc_type, CHUNK_CONFIG["generic"]).copy()

    if override_chunk_size is not None:
        cfg["chunk_size"] = override_chunk_size
    if override_chunk_overlap is not None:
        cfg["chunk_overlap"] = override_chunk_overlap

    return RecursiveCharacterTextSplitter(
        chunk_size=cfg["chunk_size"],
        chunk_overlap=cfg["chunk_overlap"],
        length_function=len,
        separators=DEFAULT_SEPARATORS,
    )

 
# Glavna funkcija za chunkovanje - prima listu dokumenata i vraća listu chunkovanih dokumenata
def chunk_documents(
    docs: List[Document],
    *,
    override_chunk_size: Optional[int] = None,
    override_chunk_overlap: Optional[int] = None,
) -> List[Document]:

    if not docs:
        return []

    all_chunks: List[Document] = []
    splitter_cache: Dict[str, RecursiveCharacterTextSplitter] = {}

    global_chunk_id = 0

    for doc in docs:

        doc_type = detect_doc_type(doc)

        if doc_type not in splitter_cache:
            splitter_cache[doc_type] = build_text_splitter_for_type(
                doc_type,
                override_chunk_size=override_chunk_size,
                override_chunk_overlap=override_chunk_overlap,
            )
        splitter = splitter_cache[doc_type]

        cleaned_content = chunk_clean(doc.page_content)
        
        base_meta = dict(doc.metadata or {})
        base_meta.setdefault("source_type", doc_type)

        cleaned_doc = Document(page_content=cleaned_content, metadata=base_meta)

        doc_chunks = splitter.split_documents([cleaned_doc])

        for idx_in_doc, ch in enumerate(doc_chunks):
            meta = dict(ch.metadata or {})
            meta.setdefault("source_type", doc_type)

            doc_id = (
                base_meta.get("doc_id")
                or base_meta.get("id")
                or base_meta.get("source_id")
            )
            if doc_id is not None:
                meta.setdefault("doc_id", doc_id)

            meta["chunk_id"] = global_chunk_id
            meta["chunk_index_in_doc"] = idx_in_doc

            ch.metadata = meta
            all_chunks.append(ch)
            global_chunk_id += 1

    return all_chunks
