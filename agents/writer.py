import re
import logging
from typing import List, Dict, Any, Callable
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AcademicWriter:
    def __init__(self, llm_invoke_func: Callable):
        """
        Writer modülü, main.py içindeki retry mantığını (invoke_gemini_chat_with_retry) 
        tekrarlamamak için bir fonksiyon referansı (callable) alır.
        """
        self.llm_invoke = llm_invoke_func
        
        self.system_prompt_template = """You are an expert software architecture technical writer. 
Your task is to write a highly professional section for an academic paper (IEEE style) based strictly on the provided codebase context.

Section Title: {section_title}
Goal of this Section: {section_goal}

CRITICAL RULES:
1. NO HALLUCINATION: Base your entire response ONLY on the provided context. If the context is insufficient, state explicitly: "Insufficient evidence in the repository to fully detail this section."
2. CITATION DISCIPLINE: You must cite the source files using the exact format: (path:start-end) or (path:lines). Example: "The authentication flow is handled by the middleware (src/auth.py:15-30)."
3. STRUCTURE: Follow this exact hierarchy:
   - Brief Introduction: Scope of the section.
   - Design/Architecture Explanation: Explain the components found in the context.
   - Security/Data Flow (if applicable): Mention any data handling or skip if not in context.
   - Diagram: You MUST include exactly ONE Mermaid diagram showing the flow. It MUST start with `graph TD`. Use only components present in the context.
   - Conclusion/Limitations: Brief wrap-up.
   
CONTEXT PROVIDED:
{context_blocks}
"""

    def _prune_code(self, content: str, max_chars: int) -> str:
        """
        Agresif budama: LLM'in token limitini şişirmemek için boş satırları ve 
        aşırı uzun blokları keser.
        """
        pruned = re.sub(r'\n\s*\n', '\n', content)
        if len(pruned) > max_chars:
            return pruned[:max_chars] + "\n...[CONTENT PRUNED FOR CONTEXT LIMITS]"
        return pruned

    def _format_context(self, parent_documents: List[Document], max_parents: int, max_chars: int) -> str:
        """
        Dökümanları standart [source=... lines=...] formatına sokar ve budar.
        (Eğer retriever.py'deki _format_context_block fonk. import edilebiliyorsa burası onunla değiştirilebilir).
        """
        context = ""
        for i, doc in enumerate(parent_documents[:max_parents]):
            path = doc.metadata.get("file_path", "unknown")
            start = doc.metadata.get("start_line", "?")
            end = doc.metadata.get("end_line", "?")
            pruned_content = self._prune_code(doc.page_content, max_chars)
            context += f"[source={path} lines={start}-{end}]\n{pruned_content}\n\n"
        return context

    def generate_section(
        self, 
        section_title: str, 
        section_goal: str, 
        parent_documents: List[Document], 
        max_parents: int = 10, 
        max_chars_per_parent: int = 6000
    ) -> Dict[str, Any]:
        """
        Orkestrasyon fonksiyonu: Veriyi alır, temizler, promptu hazırlar ve LLM'e yollar.
        """
        logging.info(f"Writer tetiklendi: {section_title} için {len(parent_documents)} döküman işleniyor.")
        
        context_str = self._format_context(parent_documents, max_parents, max_chars_per_parent)
        
        if not context_str.strip():
            return {
                "text": "Insufficient evidence in the repository to generate this section.",
                "metadata": {"parents_used": 0, "status": "no_context"}
            }

        prompt = self.system_prompt_template.format(
            section_title=section_title,
            section_goal=section_goal,
            context_blocks=context_str
        )

        try:
            logging.info("LLM'e makale taslağı için istek atılıyor...")
            response_text = self.llm_invoke(prompt) 
            
            return {
                "text": response_text,
                "metadata": {
                    "parents_used": min(len(parent_documents), max_parents),
                    "status": "success"
                }
            }
        except Exception as e:
            logging.error(f"Writer modülü LLM çağrısında hata aldı: {str(e)}")
            return {
                "text": f"Error generating section: {str(e)}",
                "metadata": {"parents_used": 0, "status": "error"}
            }