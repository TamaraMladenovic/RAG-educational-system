from pipeline.llm.factory import get_llm_adapter

def main():
    adapter = get_llm_adapter()
    llm = adapter.get_model()

    print("APP_ENV adapter:", adapter.__class__.__name__)
    print("Testing LLM...")

    response = llm.invoke("Zdravo! Samo proveravam da li radiš.").content
    print("Response:", response)

if __name__ == "__main__":
    main()