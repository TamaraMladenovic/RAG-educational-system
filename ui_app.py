from __future__ import annotations

import streamlit as st
from pipeline.rag_pipeline import RAGPipeline


@st.cache_resource
def get_rag_pipeline() -> RAGPipeline:
    return RAGPipeline()


def main() -> None:
    st.set_page_config(
        page_title="Edukativni Asistent",
        page_icon="📚",
        layout="wide",
    )

    st.title("📚 Edukativni Asistent")
    st.write(
        "Postavite pitanje iz oblasti koju učite, a sistem će koristiti "
        "informacije sa naučnih stranica i iz LAMS PDF fajlova da odgovori."
    )

    rag = get_rag_pipeline()

    # Sidebar settings (da top_k bude definisan)
    with st.sidebar:
        st.header("⚙️ Podešavanja")
        top_k = st.slider("Broj FAISS rezultata (top_k):", 1, 10, 5)

    # ✅ FORM: Enter submit + dugme submit
    with st.form(key="rag_form", clear_on_submit=False):
        question = st.text_area(
            "Unesite Vaše pitanje (na bilo kom jeziku):",
            value="",
            height=100,
        )
        submit = st.form_submit_button("Postavite pitanje")

    if submit:
        if not question.strip():
            st.warning("Unesite pitanje pre nego što pokrenete upit.")
            return

        with st.spinner("Razmišljam..."):
            result = rag.run(question, top_k=top_k)

        # Prikaz glavnog odgovora
        st.subheader("🧠 Odgovor")
        st.write(result.get("final_answer", ""))

        # Sekcija sa kontekstom / izvorima
        st.markdown("---")
        st.subheader("🔍 Korišćeni izvori")

        # Live izvori
        live_results = result.get("live_results", {})
        with st.expander("🌐 Live izvori (API pretraga)", expanded=False):
            if not live_results:
                st.write("_Nema live rezultata._")
            else:
                for source, docs in live_results.items():
                    st.markdown(f"**Izvor:** `{source}` — {len(docs)} rezultata")
                    for i, doc in enumerate(docs[:3]):
                        meta = doc.metadata or {}
                        title = meta.get("title") or meta.get("headline") or f"Dokument {i+1}"
                        st.markdown(f"- **{title}**")
                    st.markdown("---")

        # FAISS chunkovi
        retrieved_chunks = result.get("retrieved_chunks", [])
        with st.expander("📦 FAISS indeksirani chunkovi", expanded=False):
            if not retrieved_chunks:
                st.write("_Nema rezultata iz FAISS indeksa (možda još nisi ingestovao dokumente?)._")
            else:
                for ch in retrieved_chunks:
                    st.markdown(
                        f"**Doc:** `{ch.get('doc_id')}` | "
                        f"Source: `{ch.get('source')}` | "
                        f"Chunk: `{ch.get('chunk_id')}` | "
                        f"Dist: `{ch.get('distance', 0.0):.4f}`"
                    )
                    st.write(ch.get("text", ""))
                    st.markdown("---")


if __name__ == "__main__":
    main()
