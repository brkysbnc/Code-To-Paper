import re
import logging
from typing import List, Dict, Any, Callable
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class AcademicWriter:
    def __init__(self, llm_invoke_func: Callable):
        self.llm_invoke = llm_invoke_func
        
        self.system_prompt = """You are an expert software architecture technical writer. 
Write a highly professional section for an academic paper (IEEE style) based strictly on the provided context.

Section Title: {section_title}
Goal of this Section: {section_goal}

CRITICAL RULES:
1. NO HALLUCINATION: Base your response ONLY on the provided context. If context is empty or irrelevant, state: "Insufficient evidence in the repository to fully detail this section."
2. CITATION: Cite source files exactly like this: (path:start-end) or (path:lines). Example: (src/main.py:10-25).
3. STRUCTURE:
   - Brief Introduction
   - Design / Architecture Explanation
   - Security / Data Flow (if applicable, else skip)
   - Diagram: ONE Mermaid diagram starting with `graph TD`. Use only components from the context.
   - Conclusion / Limitations
4. DO NOT invent libraries or features not present in the context.

CONTEXT:
{context_blocks}
"""

    def _prune_content(self, text: str, max_chars: int) -> str:
        """Token limitini korumak için boş satırları ve sadece '#' içeren yorumları siler."""
        lines = text.split('\n')
        pruned_lines = [line for line in lines if line.strip() and not line.strip().startswith('#')]
        pruned = '\n'.join(pruned_lines)
        
        if len(pruned) > max_chars:
            return pruned[:max_chars] + "\n...[CONTENT PRUNED]"
        return pruned

    def _format_context(self, docs: List[Document], max_parents: int, max_chars: int) -> str:
        context_str = ""
        for doc in docs[:max_parents]:
            path = doc.metadata.get("file_path", "unknown")
            start = doc.metadata.get("start_line", "?")
            end = doc.metadata.get("end_line", "?")
            
            cleaned_text = self._prune_content(doc.page_content, max_chars)
            context_str += f"[source={path} lines={start}-{end}]\n{cleaned_text}\n\n"
            
        return context_str

    def generate_section(
        self, 
        section_title: str, 
        section_goal: str, 
        parent_documents: List[Document], 
        max_parents: int = 10, 
        max_chars_per_parent: int = 6000
    ) -> Dict[str, Any]:
        
        context_str = self._format_context(parent_documents, max_parents, max_chars_per_parent)
        
        if not context_str.strip():
            logger.warning("Writer tetiklendi ama context boş.")
            return {
                "text": "Insufficient evidence in the repository to fully detail this section.",
                "metadata": {"parents_used": 0, "status": "no_context"}
            }

        prompt = self.system_prompt.format(
            section_title=section_title,
            section_goal=section_goal,
            context_blocks=context_str
        )

        try:
            logger.info(f"Writer LLM'e istek atıyor: {section_title}")
            response_text = self.llm_invoke(prompt)
            return {
                "text": response_text,
                "metadata": {
                    "parents_used": min(len(parent_documents), max_parents),
                    "status": "success"
                }
            }
        except Exception as e:
            logger.error(f"Writer LLM hatası: {str(e)}")
            return {
                "text": f"Error generating section: {str(e)}",
                "metadata": {"parents_used": 0, "status": "error"}
            }