import logging
from typing import Any, Callable, Dict, List

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class AcademicWriter:
    """
    Retrieval'dan gelen parent dokumanlari kullanarak IEEE tarzinda Ingilizce bolum uretir.

    llm_invoke_func: prompt string alir, model cevabini str olarak dondurur (retry disarida).
    """

    def __init__(self, llm_invoke_func: Callable[[str], str]):
        self.llm_invoke = llm_invoke_func

        # Govde metninde path/satir yasak; kanit izlenebilirligi TRACEABILITY tablosunda.
        self.system_prompt = """You are an expert software architecture technical writer.
Write ONE highly professional section for an academic paper (IEEE style) based strictly on the provided context.

Section Title: {section_title}
Goal of this Section: {section_goal}

{repository_instructions}

OPERATOR_ADDENDUM (optional human notes; must not contradict CRITICAL RULES):
{operator_addendum}

USER_LITERATURE_APPROVED (optional labeled excerpts; implementation facts still come from CONTEXT only):
{user_literature_block}

CRITICAL RULES:
1. NO HALLUCINATION: Base repository implementation claims ONLY on CONTEXT. For claims grounded strictly in USER_LITERATURE_APPROVED, you may use only what appears there. If neither source supports a claim, write exactly: [Insufficient evidence] for that part (or the whole section if appropriate).
2. CITATION (BODY — STRICT): In the reader-facing BODY prose you must NOT include filenames, paths, line numbers, backticks around paths, or patterns like file:, lines:, (path:start-end). The ONLY inline citation markers in the BODY are numeric: [1], [2], …
3. Citation meaning: [1] = the GitHub repository as a whole (implementation), when a repository URL is provided below. Numbers [2], [3], … refer ONLY to sources explicitly labeled [2], [3], … inside USER_LITERATURE_APPROVED when that block is not "(none)". Do not use [2] for individual repository files from CONTEXT; those remain cited collectively as [1] where appropriate.
4. STRUCTURE (Markdown headings unless your training template conflicts):
### Scope
### Design and implementation (from code)
### Security / trust boundaries (only if evidenced in CONTEXT; otherwise one honest paragraph, may use [Insufficient evidence])
### Mermaid snapshot
### Limitations
Exception — when Section Title is exactly "Literature Review" or exactly "Related Work": do NOT use the generic Scope/Design/Security/Mermaid/Limitations ### template. Use ### subheadings that reflect actual cited works or distinct themes grounded in USER_LITERATURE_APPROVED and CONTEXT (e.g. one ### per major cited source or grouping), not placeholder generic section names.
Exception — when Section Title is exactly "Introduction" or exactly "Conclusion": omit "### Mermaid snapshot" entirely and include NO fenced Mermaid diagram anywhere in PART 1 — PAPER BODY (rules 4–5 on Mermaid do not apply to those two section titles).
5. MERMAID: Exactly ONE fenced Mermaid block (unless Section Title is Introduction or Conclusion per rule 4 exceptions); first line inside the fence MUST be: graph TD. Use only components/flows evidenced in CONTEXT.
6. DO NOT invent libraries, services, or features not present in the context.

OUTPUT — IN THIS ORDER (labels help downstream parsing):
PART 1 — PAPER BODY
(Prose + ### headings + single Mermaid; [n] citations only; no paths/lines.)

PART 2 — REFERENCES
(Short IEEE-like list; [1] MUST use the repository URL when provided below; do not invent URLs.
If USER_LITERATURE_APPROVED contained [2], [3], … include concise entries for those user-provided sources without fabricating publishers or URLs.)

PART 3 — TRACEABILITY
Start this subsection with a line containing only:
TRACEABILITY:
Then a markdown table with columns:
| Claim ID | Claim summary | Source file | Lines | Notes |
Map substantive BODY claims to CONTEXT locations (repository code). Paths/lines appear ONLY here and in CONTEXT, not in PART 1. For claims supported only by USER_LITERATURE_APPROVED, use Source file = `User literature`, Lines = `n/a`, and cite the matching [n] in Notes.

CONTEXT:
{context_blocks}
"""

    def _prune_content(self, text: str, max_chars: int) -> str:
        """Token limitini korumak icin bos satirlari ve sadece '#' iceren yorum satirlarini siler."""
        lines = text.split("\n")
        pruned_lines = [line for line in lines if line.strip() and not line.strip().startswith("#")]
        pruned = "\n".join(pruned_lines)

        if len(pruned) > max_chars:
            return pruned[:max_chars] + "\n...[CONTENT PRUNED]"
        return pruned

    def _format_context(self, docs: List[Document], max_parents: int, max_chars: int) -> str:
        """Parent dokumanlari Writer promptuna uygun tek metin blogunda birlestirir."""
        context_str = ""
        for doc in docs[:max_parents]:
            path = doc.metadata.get("file_path", "unknown")
            start = doc.metadata.get("start_line", "?")
            end = doc.metadata.get("end_line", "?")

            cleaned_text = self._prune_content(doc.page_content or "", max_chars)
            context_str += f"[source={path} lines={start}-{end}]\n{cleaned_text}\n\n"

        return context_str

    def _repository_instruction_block(self, repository_url: str) -> str:
        """
        [1] kaynagi icin modele net talimat uretir; URL yoksa uydurma yapmamasi soylenir.
        """
        url = (repository_url or "").strip()
        if url:
            web = url[:-4] if url.lower().endswith(".git") else url
            return (
                "REPOSITORY FOR CITATION [1]:\n"
                f"- Use this URL in PART 2 — References for [1] (web form preferred): {web}\n"
                "- In PART 1 body, cite the codebase as [1] where appropriate; do not paste raw URLs in the body unless IEEE explicitly requires it."
            )
        return (
            "REPOSITORY FOR CITATION [1]:\n"
            "- No repository URL was supplied. Do NOT invent a URL. In PART 2, you may title [1] generically "
            '("Analyzed software repository") without an "Available" URL line, or state that the URL was not provided.'
        )

    @staticmethod
    def _split_traceability(raw: str) -> tuple[str, str]:
        """
        LLM ciktisinda yalnizca 'TRACEABILITY:' satirindan sonraki izleme tablosunu ayirir.
        Aksi halde 'PART 3 — TRACEABILITY:' gibi basliklar yanlis eslesirdi.
        """
        lines = raw.splitlines()
        for i, line in enumerate(lines):
            if line.strip() == "TRACEABILITY:":
                body = "\n".join(lines[:i]).strip()
                tail = "\n".join(lines[i:]).strip()
                return body, tail
        return raw.strip(), ""

    def generate_section(
        self,
        section_title: str,
        section_goal: str,
        parent_documents: List[Document],
        max_parents: int = 10,
        max_chars_per_parent: int = 6000,
        *,
        repository_url: str = "",
        operator_addendum: str = "",
        user_literature_block: str = "",
    ) -> Dict[str, Any]:
        """Baglamdan bolum metni, istege bagli TRACEABILITY parcasi ve metadata uretir."""
        context_str = self._format_context(parent_documents, max_parents, max_chars_per_parent)

        if not context_str.strip():
            logger.warning("Writer tetiklendi ama context bos.")
            return {
                "text": "Insufficient evidence in the repository to fully detail this section.",
                "metadata": {"parents_used": 0, "status": "no_context", "traceability": ""},
            }

        repo_block = self._repository_instruction_block(repository_url)
        op_raw = (operator_addendum or "").strip()
        op_block = op_raw if op_raw else "(none — follow only built-in rules.)"
        lit_raw = (user_literature_block or "").strip()
        lit_block = lit_raw if lit_raw else "(none — do not invent external literature citations.)"

        # str.format kullanilmaz; context'teki { } kod bloklari bozulmaz. Yer tutucular sondan basa dogru da degil,
        # degerlerde baska yer tutucu metni yoksa sirasi onemsiz; context en sonda degistirilir.
        prompt = self.system_prompt
        prompt = prompt.replace("{section_title}", section_title)
        prompt = prompt.replace("{section_goal}", section_goal)
        prompt = prompt.replace("{repository_instructions}", repo_block)
        prompt = prompt.replace("{operator_addendum}", op_block)
        prompt = prompt.replace("{user_literature_block}", lit_block)
        prompt = prompt.replace("{context_blocks}", context_str)

        try:
            logger.info("Writer LLM'e istek atiyor: %s", section_title)
            response_text = self.llm_invoke(prompt)
            body_text, trace_text = self._split_traceability(response_text)
            return {
                "text": body_text if trace_text else response_text,
                "metadata": {
                    "parents_used": min(len(parent_documents), max_parents),
                    "status": "success",
                    "traceability": trace_text,
                    **({"full_response": response_text} if trace_text else {}),
                },
            }
        except Exception as e:  # noqa: BLE001
            logger.error("Writer LLM hatasi: %s", e)
            return {
                "text": f"Error generating section: {e}",
                "metadata": {"parents_used": 0, "status": "error", "traceability": ""},
            }
