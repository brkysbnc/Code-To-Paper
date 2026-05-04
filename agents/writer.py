import logging
import re
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

CRITICAL — NO LITERATURE-TO-REPOSITORY TERMINOLOGY BLEED:
- USER_LITERATURE_APPROVED summarizes OTHER people's systems (e.g. a cited thesis). Their worker names, datasets, and industrial domains belong ONLY to those cited sources ([2], …). Never imply they are names of components inside [1].
- When describing what THIS repository ([1]) implements, use ONLY vocabulary evidenced in CONTEXT: real file paths, Python module/agent class names, README phrases, and architecture words that literally appear there. Do NOT rename [1]'s pipeline using cool labels copied from literature unless CONTEXT contains those exact strings.
- Forbidden as labels for [1] unless they literally appear in CONTEXT (as identifiers or quoted docs): phrases such as "Document Reader", "Map Extractor", "Section Writer", "Report Assembler", "Abstract Generator Node", "Header Generator Node", "Update Vector Database Node", "Acronym Extractor Node" — these often appear in cited multi-agent report papers but are NOT generic IEEE terms; they are thesis-specific roles.
- Prefer CONTEXT-grounded wording instead, for example: GitHub ingestion / clone step, indexer or parent-child chunks in Chroma, planner-generated retrieval queries, AcademicWriter / MetadataWriter agents, faithfulness judge / verification step, Streamlit UI, LangChain stack, Gemini models — only where CONTEXT supports each phrase.
- Literature Review paragraph (EXISTING WORK) may use the cited sources' own terminology WITH attribution ([2], …). Paragraphs THE GAP and OUR CONTRIBUTION still follow CRITICAL RULES 1–3: OUR CONTRIBUTION must describe [1] strictly from CONTEXT without borrowing foreign worker-node branding.

DOMAIN LOCK (Introduction, Methodology, System Architecture and Implementation, Conclusion — NOT for Literature Review EXISTING WORK):
- Do NOT frame [1] as truck testing, Volvo fleets, expedition logbooks, field Engineering Reports from vehicles, or similar domains unless CONTEXT explicitly discusses those domains for this repository.
- Default framing for Code-To-Paper–style repos: automation of IEEE/academic-style manuscript preparation from GitHub source plus optional literature grounding (RAG, LLM orchestration, verification).

4. STRUCTURE: Write flowing academic prose. Use ### subheadings ONLY
   when the content genuinely requires subdivision into distinct topics.
   You are NOT required to use any fixed subheading names.
   - Choose subheading titles that accurately describe the specific
     content you are writing, based solely on what CONTEXT supports.
   - Do NOT use these generic placeholder names: "Scope",
     "Design and implementation", "Security / trust boundaries",
     "Mermaid snapshot", "Limitations" — unless those exact topics
     are the most accurate description of what CONTEXT contains.
   - Do NOT add a subheading just to create structure. If the content
     flows naturally as continuous prose, write it that way.
   - Subheadings must be specific to this project's actual components
     (e.g. "Parent-Child Indexing Strategy", "Adaptive Retrieval
     Threshold", "Rate-Limited Embedding Pipeline").
   Exception — Introduction and Conclusion: write as flowing prose,
   no ### subheadings at all, no Mermaid diagram.
   Exception — when Section Title is exactly 'Literature Review'
   or exactly 'Related Work':
     Write THREE paragraphs in this exact order:
     1) EXISTING WORK: Summarize each cited source from
        USER_LITERATURE_APPROVED in 2-3 sentences. Cover: problem
        they addressed, method used, result achieved. Use [n] citations.
        Do NOT invent. If a source says nothing about a topic, skip it.
     2) THE GAP: State what the cited works collectively do NOT address,
        grounded only in what USER_LITERATURE_APPROVED contains.
        Be specific — name the missing capability (e.g. 'None of the
        cited works address automated IEEE-format generation from
        GitHub source code repositories').
     3) OUR CONTRIBUTION: Explain how the analyzed repository [1]
        addresses the gap. Connect to specific implementation details
        from CONTEXT (e.g. parent-child chunking in Chroma, multi-query
        retrieval, faithfulness judge). Cite [1] for implementation.
        Do NOT claim features not in CONTEXT.
     OPTIONAL TABLE: You MAY add a comparison table, but ONLY under ALL of the following conditions:
     1. There are 2 or more cited works in USER_LITERATURE_APPROVED.
     2. You can fill at least 3 rows with real data directly from USER_LITERATURE_APPROVED and CONTEXT — no invented data.
     3. CRITICAL — ATOMIC RULE: You must decide whether to include the table BEFORE writing any ### subheading. If you are not certain you can satisfy conditions 1 and 2, do NOT write the ### subheading at all. Writing a ### subheading and then omitting the table is strictly forbidden. The subheading and the table are a single unit: either both appear or neither appears.
     NO Mermaid diagram in Literature Review.
     ABSOLUTE PROHIBITION: Do NOT write any ### subheading whose
     text matches or closely resembles the section title
     'Literature Review' or 'Related Work'. This includes variations
     like 'A. Literature Review', 'Overview of Literature', or
     'Literature and Background'. Violating this rule makes the
     document look broken.
     The THREE paragraphs (EXISTING WORK, THE GAP, OUR CONTRIBUTION)
     must be written as continuous prose with NO ### subheadings
     between them. Only add a ### subheading if you are inserting
     a comparison TABLE, and in that case the subheading must
     describe the table content specifically (e.g.
     '### Comparison of Automated Report Generation Approaches').

4b. SUBHEADING FORMAT: When you write ### subheadings, write ONLY
    the title text. Do NOT prefix with A. B. C. or Roman numerals.
    Do NOT write "### A. Pipeline Design" — write "### Pipeline Design".
    The export layer adds letter prefixes automatically.

4c. SUBHEADING UNIQUENESS: The FIRST ### subheading of any section
     MUST NOT repeat or paraphrase the section title.
     - If Section Title is 'System architecture and implementation',
       do NOT write '### System Architecture and Implementation' or
       '### System Overview' as the first subheading.
     - If Section Title is 'Literature Review', do NOT write
       '### Literature Review' or '### Related Work' as any subheading.
     - Instead, the first subheading must immediately name a SPECIFIC
       technical component or theme (e.g. '### Repository Ingestion
       Pipeline', '### Parent-Child Chunking Strategy').
     - If you cannot think of a specific subheading that adds value
       beyond the section title, write the content as prose without
       any subheading.

5. MERMAID: You MAY include one optional fenced Mermaid diagram
   (first line inside fence MUST be: graph TD) ONLY IF the diagram
   adds meaningful architectural insight not already clear from prose.
   Use only components/flows directly evidenced in CONTEXT.
   If the section does not benefit from a diagram, omit it entirely.
   Exception — Introduction and Conclusion: never include Mermaid.

6. DO NOT invent libraries, services, or features not present in the context.
7. ANTI-REPETITION (Introduction only): When writing Introduction,
   you will NOT have access to the abstract text, but assume one exists.
   The Introduction MUST NOT start with 'The rapid', 'This paper',
   'In recent years', 'The growing', or any other cliché academic
   opening. Instead:
   - Open with a CONCRETE PROBLEM STATEMENT grounded in CONTEXT
     (e.g. a specific gap, a specific pain point visible in the code).
   - The Introduction must cover motivation, scope, and structure
     of the paper — NOT repeat what an abstract would say.
   - Follow Section Goal paragraph structure when provided (typically four blocks:
     problem; background/motivation; explicit contributions sentence starting with
     'The contributions of this paper are:'; final roadmap paragraph).
   - Put the section-by-section roadmap ONLY in that final roadmap paragraph,
     listing ONLY sections that exist in this manuscript (per Section Goal).
     Do NOT place the full roadmap in the second paragraph.

OUTPUT — IN THIS ORDER (labels help downstream parsing):
PART 1 — PAPER BODY
(Prose + ### headings + single Mermaid; [n] citations only; no paths/lines.)

PART 2 — REFERENCES
(Short IEEE-like list; [1] MUST use the repository URL when provided below; do not invent URLs.
For [1] (the repository): the author is the GitHub username or profile name from the repository URL, NOT any name from USER_LITERATURE_APPROVED. Do not assign literature authors to the repository citation. If the owner name is unclear from the URL, write the GitHub username as-is.
If USER_LITERATURE_APPROVED contained [2], [3], … include concise entries for those user-provided sources without fabricating publishers or URLs.)

PART 3 — TRACEABILITY
Start this subsection with a line containing only:
TRACEABILITY:
Then a markdown table with columns:
| Claim ID | Claim summary | Source file | Lines | Notes |

CRITICAL — SOURCE FILE RULES FOR TRACEABILITY TABLE:
- If the claim is based on code you read in CONTEXT:
  Source file = the actual file path (e.g. agents/writer.py)
- If the claim is based on USER_LITERATURE_APPROVED:
  Source file = User literature   ← EXACTLY these two words
  Lines = n/a
  WRONG examples (never do this):
    Source file = makale1.pdf
    Source file = [2]
    Source file = external
  CORRECT example:
    | C3 | RAG improves grounding | User literature | n/a | See [2] |
- If you are not sure which source supports a claim, write
  [Insufficient evidence] in the body and omit the claim from TRACEABILITY.

Map substantive BODY claims to CONTEXT or USER_LITERATURE_APPROVED only. Paths/lines appear ONLY here and in CONTEXT, not in PART 1.

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
        LLM ciktisinda 'TRACEABILITY:' marker satirini bulup sonraki tabloyu ayirir.

        Esnek match: preview modelin uretebilecegi varyantlari da kabul eder:
          - 'TRACEABILITY:'
          - '**TRACEABILITY:**'  (markdown bold)
          - '## TRACEABILITY'    (markdown header)
          - 'PART 3 — TRACEABILITY:' (prefix korumali)
          - 'TRACEABILITY' (sondaki ':' eksik)
        Body icinde gecen 'traceability' kelimesi ile false-positive vermesin diye
        marker yalniz tek satir olmali (line.strip() sonrasi sadece TRACEABILITY).
        """
        lines = raw.splitlines()
        for i, line in enumerate(lines):
            clean = re.sub(r"^[\s#*_`\-]+", "", line)
            clean = re.sub(r"^PART\s*\d+\s*[—\-–:]\s*", "", clean, flags=re.IGNORECASE)
            clean = re.sub(r"[\s*_`]+$", "", clean.strip())
            if re.match(r"^TRACEABILITY\s*:?\s*$", clean, flags=re.IGNORECASE):
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
