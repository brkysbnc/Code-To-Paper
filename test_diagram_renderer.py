import logging, os
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from export.diagram_renderer import generate_all_diagrams

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
load_dotenv()

SAMPLE_CONTEXT = """
agents/writer.py: AcademicWriter class with generate_section() method using LLM.
agents/faithfulness_judge.py: judge_section_faithfulness() function.
orchestration/section_pipeline.py: run_paper_pipeline().
export/ieee_template_export.py: markdown_to_ieee_template_docx_bytes().
Stack: Python, LangChain, ChromaDB, Google Gemini, python-docx, Streamlit.
"""

def _invoke(prompt: str) -> str:
    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY", ""), temperature=0.2)
    return llm.invoke(prompt).content

if __name__ == "__main__":
    results = generate_all_diagrams(SAMPLE_CONTEXT, _invoke)
    passed = sum(1 for p in results.values() if p and Path(p).exists())
    for t, p in results.items():
        print(f"{'✓' if p and Path(p).exists() else '✗'} {t}: {p or 'FAILED'}")
    print(f"\nTest results: {passed}/3 passed")
