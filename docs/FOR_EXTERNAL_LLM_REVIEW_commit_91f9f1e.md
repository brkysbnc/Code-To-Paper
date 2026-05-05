# Harici LLM incelemesi: commit `91f9f1e` (SOURCE SEPARATION refactor)

Bu dosya **Gemini / ChatGPT** gibi araçlara yapıştırmak içindir. Türkçe özet + teknik içerik İngilizce tutulmuştur.

## Türkçe özet (senin için)

- **Baseline:** `79cde2e` — bu sürümde üretilen Word çıktısı (`commitedonus.docx`) **Introduction düzgün** (düz paragraf).
- **Sonra:** `91f9f1e` cherry-pick sonrası (`88d8e29`) üretilen Word (`committensonra1.docx`) — **Introduction başlığından hemen sonra tam makale JSON’u** düşüyor (`{`, `"title"`, `"sections"` …).
- **Bu commit’te değişen dosyalar:** `agents/writer.py` (AcademicWriter `system_prompt`), `orchestration/paper_blueprint.py` (`DEFAULT_PAPER_SECTIONS` hedef metinleri).

Aşağıda **git unified diff** var: soldaki `-` satırları **eski** (`79cde2e`), sağdaki `+` satırları **yeni** (`91f9f1e` / şu anki HEAD).

---

## Observed failure (paste into external LLM)

**Symptom:** After upgrading prompts from commit baseline `79cde2e` to patch equivalent to `91f9f1e`, the generated DOCX places a **full IEEE-style JSON document** immediately under the Heading-1 **“Introduction and Motivation”**, instead of normal prose. Subsequent sections (e.g. Literature Review) may render as normal Markdown-derived prose below that JSON blob.

**Example structure in broken DOCX:**

- Paragraph: `I. Introduction and Motivation`
- Next paragraphs: `{`, `"title": "...", "authors": [...], "sections": [...]` …

**Hypothesis buckets for reviewers:**

1. **Indirect prompt injection:** Retrieved CONTEXT contains another agent file (e.g. `IEEE_JSON_ONLY_PROMPT` string in `agents/ieee_json_writer.py`). Could the **rewritten SOURCE SEPARATION block** interact badly with model behavior (weaker anchoring to “output prose PART 1”, more imitation of structured schemas visible in CONTEXT)?
2. **Prompt regression:** Removal of concrete examples (“AcademicWriter”, “faithfulness judge”, “Chroma”, explicit DOMAIN LOCK examples) reduces formatting discipline.
3. **paper_blueprint section goals:** Less prescriptive Methodology/System text might change planner/writer trajectory — unlikely alone to force JSON unless combined with retrieval.
4. **Coincidence / nondeterminism:** Same codebase sometimes differs by retrieval ordering; user reports reproducible regression tied to this commit — treat as priority.

---

## Unified diff (OLD = `79cde2e`, NEW = `HEAD` after cherry-pick)

```diff
diff --git a/agents/writer.py b/agents/writer.py
index 6d23856..80c2d08 100644
--- a/agents/writer.py
+++ b/agents/writer.py
@@ -37,16 +37,15 @@ CRITICAL RULES:
 2. CITATION (BODY — STRICT): In the reader-facing BODY prose you must NOT include filenames, paths, line numbers, backticks around paths, or patterns like file:, lines:, (path:start-end). The ONLY inline citation markers in the BODY are numeric: [1], [2], …
 3. Citation meaning: [1] = the GitHub repository as a whole (implementation), when a repository URL is provided below. Numbers [2], [3], … refer ONLY to sources explicitly labeled [2], [3], … inside USER_LITERATURE_APPROVED when that block is not "(none)". Do not use [2] for individual repository files from CONTEXT; those remain cited collectively as [1] where appropriate.
 
-CRITICAL — NO LITERATURE-TO-REPOSITORY TERMINOLOGY BLEED:
-- USER_LITERATURE_APPROVED summarizes OTHER people's systems (e.g. a cited thesis). Their worker names, datasets, and industrial domains belong ONLY to those cited sources ([2], …). Never imply they are names of components inside [1].
-- When describing what THIS repository ([1]) implements, use ONLY vocabulary evidenced in CONTEXT: real file paths, Python module/agent class names, README phrases, and architecture words that literally appear there. Do NOT rename [1]'s pipeline using cool labels copied from literature unless CONTEXT contains those exact strings.
-- Forbidden as labels for [1] unless they literally appear in CONTEXT (as identifiers or quoted docs): phrases such as "Document Reader", "Map Extractor", "Section Writer", "Report Assembler", "Abstract Generator Node", "Header Generator Node", "Update Vector Database Node", "Acronym Extractor Node" — these often appear in cited multi-agent report papers but are NOT generic IEEE terms; they are thesis-specific roles.
-- Prefer CONTEXT-grounded wording instead, for example: GitHub ingestion / clone step, indexer or parent-child chunks in Chroma, planner-generated retrieval queries, AcademicWriter / MetadataWriter agents, faithfulness judge / verification step, Streamlit UI, LangChain stack, Gemini models — only where CONTEXT supports each phrase.
-- Literature Review paragraph (EXISTING WORK) may use the cited sources' own terminology WITH attribution ([2], …). Paragraphs THE GAP and OUR CONTRIBUTION still follow CRITICAL RULES 1–3: OUR CONTRIBUTION must describe [1] strictly from CONTEXT without borrowing foreign worker-node branding.
+SOURCE SEPARATION — REPOSITORY [1] VS CITED LITERATURE [2]+ (no keyword lists; applies to ANY repo and ANY uploaded papers):
+- CONTEXT is the ONLY evidence base for WHAT [1] IS AND DOES: modules, pipelines, services, frameworks named in code/docs, dependencies, APIs, databases, UI layers, deployment hints, stated goals, evaluation hooks—anything asserted as fact about THIS repository must be supported by CONTEXT (paraphrase allowed; invented components forbidden).
+- USER_LITERATURE_APPROVED is the ONLY evidence base for WHAT EXTERNAL SOURCES SAY about THEIR OWN work when cited as [2], [3], … Never merge their proprietary labeling into [1].
+- NEVER transplant vocabulary from USER_LITERATURE_APPROVED (subsystem titles, coined agent/job names, dataset or product names, application-domain vignettes from cited PDFs) into sentences that describe [1]'s architecture or behavior—unless that EXACT wording also appears inside CONTEXT as part of THIS repo's docs/code (then it is CONTEXT-grounded, not literature-imported).
+- Contrasts are allowed ONLY when attribution is explicit: attribute foreign claims with [n]; describe [1] with CONTEXT-backed facts in the same sentence or adjacent sentences. Do not silently reuse another paper's component nouns as names for [1]'s pieces.
+- Literature Review — EXISTING WORK: use each cited source's own terms WITH [n]. THE GAP: grounded in USER_LITERATURE_APPROVED only. OUR CONTRIBUTION: describe how [1] fills the gap using CONTEXT-derived naming ONLY for [1]'s mechanisms.
 
-DOMAIN LOCK (Introduction, Methodology, System Architecture and Implementation, Conclusion — NOT for Literature Review EXISTING WORK):
-- Do NOT frame [1] as truck testing, Volvo fleets, expedition logbooks, field Engineering Reports from vehicles, or similar domains unless CONTEXT explicitly discusses those domains for this repository.
-- Default framing for Code-To-Paper–style repos: automation of IEEE/academic-style manuscript preparation from GitHub source plus optional literature grounding (RAG, LLM orchestration, verification).
+APPLICATION DOMAIN FOR [1] (Introduction, Methodology, System Architecture and Implementation, Conclusion — NOT Literature Review EXISTING WORK):
+- Problem domain, users, setting, data modality, and industry scenario for [1] MUST be inferred ONLY from CONTEXT (README, configs, identifiers, comments). Do NOT adopt application-domain framing from USER_LITERATURE_APPROVED when describing [1], unless CONTEXT explicitly ties [1] to that domain.
 
 4. STRUCTURE: Write flowing academic prose. Use ### subheadings ONLY
    when the content genuinely requires subdivision into distinct topics.
@@ -59,9 +58,8 @@ DOMAIN LOCK (Introduction, Methodology, System Architecture and Implementation,
      are the most accurate description of what CONTEXT contains.
    - Do NOT add a subheading just to create structure. If the content
      flows naturally as continuous prose, write it that way.
-   - Subheadings must be specific to this project's actual components
-     (e.g. "Parent-Child Indexing Strategy", "Adaptive Retrieval
-     Threshold", "Rate-Limited Embedding Pipeline").
+   - Subheadings must describe specifics evidenced in CONTEXT for THIS repo
+     (avoid recycled titles from unrelated papers).
    Exception — Introduction and Conclusion: write as flowing prose,
    no ### subheadings at all, no Mermaid diagram.
    Exception — when Section Title is exactly 'Literature Review'
@@ -78,8 +76,8 @@ DOMAIN LOCK (Introduction, Methodology, System Architecture and Implementation,
         GitHub source code repositories').
      3) OUR CONTRIBUTION: Explain how the analyzed repository [1]
         addresses the gap. Connect to specific implementation details
-        from CONTEXT (e.g. parent-child chunking in Chroma, multi-query
-        retrieval, faithfulness judge). Cite [1] for implementation.
+        from CONTEXT only (cite mechanisms visible in retrieved chunks).
+        Cite [1] for implementation.
         Do NOT claim features not in CONTEXT.
      OPTIONAL TABLE: You MAY add a comparison table, but ONLY under ALL of the following conditions:
      1. There are 2 or more cited works in USER_LITERATURE_APPROVED.
diff --git a/orchestration/paper_blueprint.py b/orchestration/paper_blueprint.py
index f510c0b..06ff81b 100644
--- a/orchestration/paper_blueprint.py
+++ b/orchestration/paper_blueprint.py
@@ -162,30 +162,26 @@ DEFAULT_PAPER_SECTIONS: list[tuple[str, str]] = [
         "they leave uncovered, then explain how the analyzed repository "
         "fills that gap using only evidence from CONTEXT and "
         "USER_LITERATURE_APPROVED. Do not invent claims. "
-        "In OUR CONTRIBUTION prose, describe [1] using CONTEXT terminology only — "
-        "do not reuse cited papers' worker-node names as labels for this repository unless CONTEXT contains them verbatim. "
+        "In OUR CONTRIBUTION prose, name [1]'s mechanisms ONLY from CONTEXT; "
+        "do not reuse cited papers' subsystem or agent naming schemes for [1] unless those strings literally appear in CONTEXT. "
         "Write as flowing prose with NO subheadings.",
     ),
     (
         "Methodology",
-        "Describe the research methodology and system design approach. "
-        "Explain how the repository addresses the problem: the overall pipeline strategy, "
-        "the parent-child chunking approach for document indexing, "
-        "the planner-driven multi-query retrieval mechanism, "
-        "and the faithfulness validation process. "
-        "Focus on the WHY and HOW of design decisions, not the implementation details. "
-        "Name pipeline stages ONLY with CONTEXT-evidenced terms; do not borrow worker-node titles from USER_LITERATURE_APPROVED for [1]. "
+        "Describe the methodology and design rationale that CONTEXT supports for THIS repository "
+        "(e.g. indexing or retrieval choices, orchestration, verification, generation workflow—ONLY topics evidenced in CONTEXT). "
+        "Focus on WHY/HOW where CONTEXT explains intent or structure; omit topics absent from CONTEXT. "
+        "Every named stage or architectural element attributed to [1] must be CONTEXT-derived; "
+        "never lift subsystem labels from user-supplied literature onto [1]. "
         "Ground all claims strictly in repository evidence.",
     ),
     (
         "System Architecture and Implementation",
-        "Detail the full technical architecture and implementation. "
-        "Explain repository ingestion, hierarchical indexing with ChromaDB, "
-        "planner-driven multi-query retrieval, the academic writer stage, "
-        "and the faithfulness judge component. "
-        "Describe the technical stack: Python, Gemini models, LangChain, Streamlit. "
-        "Do not describe [1] using another paper's bespoke agent labels unless those strings appear in CONTEXT. "
-        "Map all claims strictly to repository evidence.",
+        "Detail architecture and implementation strictly from CONTEXT: components, modules, "
+        "data layers, external services, languages and frameworks mentioned in THIS repo. "
+        "Do not assume a fixed stack (any DB, LLM, UI, or agent framework)—only report what CONTEXT evidences. "
+        "Never describe [1] using naming borrowed from uploaded literature unless CONTEXT contains those strings. "
+        "Map every substantive claim to repository evidence.",
     ),
     (
         "Conclusion",
@@ -196,9 +192,9 @@ DEFAULT_PAPER_SECTIONS: list[tuple[str, str]] = [
         "(c) Limitations of the current system grounded in repository evidence "
         "(e.g. reliance on external LLM APIs, retrieval bounds, free-tier constraints). "
         "(d) Future work directions supported by the codebase. "
-        "DOMAIN: Summarize ONLY what [1] actually does per CONTEXT — typically GitHub-to-IEEE/RAG documentation automation. "
-        "Do NOT drift into cited literature's application domain (e.g. truck testing, fleet logbooks, Volvo-specific workflows) "
-        "unless CONTEXT proves this repository targets that domain. "
+        "DOMAIN: Summarize ONLY scope and outcomes that CONTEXT establishes for [1]. "
+        "Do NOT adopt scenarios, industries, datasets, or evaluation claims from uploaded literature "
+        "unless CONTEXT explicitly ties [1] to them. "
         "Do NOT write a separate Limitations section — integrate everything here. "
         "Total length: 3-4 paragraphs.",
     ),
```

---

## Önerilen soru (İngilizce, harici LLM’e kopyala)

```
We have a regression in an LLM pipeline (Gemini) that generates IEEE-style paper sections.

Before prompt edits (baseline commit), DOCX export looks correct: Introduction is normal prose.

After replacing the CRITICAL RULES block in AcademicWriter.system_prompt with SOURCE SEPARATION text AND loosening DEFAULT_PAPER_SECTIONS guidance (remove explicit tech stack anchors like ChromaDB/Gemini/LangChain), the Introduction section often becomes a pasted JSON document matching our separate ieee_json_writer schema.

CONTEXT retrieval includes repository source files; some files contain strings like “Output ONLY one valid JSON object”.

Task:
1) Which specific additions/removals in the diff could plausibly increase JSON-like outputs in the Introduction writer?
2) Propose minimal prompt guardrails to prevent imitating JSON schemas / instruction strings found in CONTEXT, without reintroducing huge hardcoded word lists.
3) Suggest experiments to confirm root cause (e.g., ablation of individual bullet changes).
```

---

## Dosya konumu

**`c:\Users\berka\Code-To-Paper\docs\FOR_EXTERNAL_LLM_REVIEW_commit_91f9f1e.md`**
