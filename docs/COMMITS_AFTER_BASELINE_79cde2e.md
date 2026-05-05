# Geri donus notu: baseline `79cde2e` sonrasi 2 commit

Bu dosya, **bu baseline'in hemen ustunde** kalan **tam 2 commit**in kelimesi kelimesine `git show --no-color` ciktisidir.

GitHub sirasi (yeniden eskiye): once `cd545b4` (Mermaid), onun altinda `91f9f1e` (SOURCE SEPARATION refactor).

## Baseline (reset / checkout hedefi)

- **Tam hash:** `79cde2e4c5dc25d0f94face8ee7ca868145f2b69`
- **Ozeti:** fix(writer): literatur terminolojisinin repoya sizmasini engelle  
- **Bu committe degisen dosyalar:** `agents/writer.py`, `orchestration/paper_blueprint.py`

## Bu dosyada kronolojik sira (yeniden uygulama: eski commit once)

| Sira | Kisa hash | Ozet |
|------|-----------|------|
| 1 | `91f9f1e` | refactor(writer): literatur sizintisi icin ilke tabanli SOURCE SEPARATION |
| 2 | `cd545b4` | fix(export): Word'e Mermaid kod sizmasini engelle; writer'da govde icin Mermaid yasagi |

**Degisen dosyalar (commit bazinda):**

1. `91f9f1e`: `agents/writer.py`, `orchestration/paper_blueprint.py`
2. `cd545b4`: `agents/writer.py`, `export/ieee_template_export.py`

## Yeniden uygulama (baseline'a dondukten sonra)

```bash
git cherry-pick 91f9f1e0dc04ed285a11a948fca81919b9b5d14e
git cherry-pick cd545b459ae798efb1905cb9b770ad6fb9a39823
```

---



## ===== COMMIT 1/2: `git show 91f9f1e` =====

````text
commit 91f9f1e0dc04ed285a11a948fca81919b9b5d14e
Author: Berkay Sabuncu <berkaysbncc@gmail.com>
Date:   Mon May 4 21:25:16 2026 +0300

    refactor(writer): literatur sizintisi icin ilke tabanli SOURCE SEPARATION
    
    - Sabit kelime yasak listesi kaldirildi; CONTEXT vs USER_LITERATURE kanit ayrimi
    
    - Alan/domain kilidi literaturden aktarma ilkesiyle genellenmiştir
    
    - paper_blueprint Methodology/System tamamen CONTEXT-evidence odakli (sabit stack varsayimi yok)
    
    Co-authored-by: Cursor <cursoragent@cursor.com>

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
````


## ===== COMMIT 2/2: `git show cd545b4` =====

````text
commit cd545b459ae798efb1905cb9b770ad6fb9a39823
Author: Berkay Sabuncu <berkaysbncc@gmail.com>
Date:   Tue May 5 01:29:57 2026 +0300

    fix(export): Word'e Mermaid kod sizmasini engelle; writer'da govde icin Mermaid yasagi
    
    Co-authored-by: Cursor <cursoragent@cursor.com>

diff --git a/agents/writer.py b/agents/writer.py
index 80c2d08..ee24c79 100644
--- a/agents/writer.py
+++ b/agents/writer.py
@@ -83,7 +83,7 @@ APPLICATION DOMAIN FOR [1] (Introduction, Methodology, System Architecture and I
      1. There are 2 or more cited works in USER_LITERATURE_APPROVED.
      2. You can fill at least 3 rows with real data directly from USER_LITERATURE_APPROVED and CONTEXT — no invented data.
      3. CRITICAL — ATOMIC RULE: You must decide whether to include the table BEFORE writing any ### subheading. If you are not certain you can satisfy conditions 1 and 2, do NOT write the ### subheading at all. Writing a ### subheading and then omitting the table is strictly forbidden. The subheading and the table are a single unit: either both appear or neither appears.
-     NO Mermaid diagram in Literature Review.
+     NO Mermaid diagram, diagram markup, or ASCII arrow-flow lines in Literature Review.
      ABSOLUTE PROHIBITION: Do NOT write any ### subheading whose
      text matches or closely resembles the section title
      'Literature Review' or 'Related Work'. This includes variations
@@ -116,12 +116,13 @@ APPLICATION DOMAIN FOR [1] (Introduction, Methodology, System Architecture and I
        beyond the section title, write the content as prose without
        any subheading.
 
-5. MERMAID: You MAY include one optional fenced Mermaid diagram
-   (first line inside fence MUST be: graph TD) ONLY IF the diagram
-   adds meaningful architectural insight not already clear from prose.
-   Use only components/flows directly evidenced in CONTEXT.
-   If the section does not benefit from a diagram, omit it entirely.
-   Exception — Introduction and Conclusion: never include Mermaid.
+5. MERMAID / DIYAGRAM METNI (KESIN YASAK): Do NOT include Mermaid syntax
+   in PART 1 under any circumstances — no ```mermaid fences, no ``` code
+   blocks that contain graph TD / flowchart / classDiagram / erDiagram,
+   no fenceless lines like 'graph TD', and no ASCII flow lines such as
+   'A[Label] --> B[Label]'. Architectural figures for the paper are produced
+   separately by the toolchain as PNG placeholders ([DIAGRAM:…]); describe
+   architecture in prose only. Violating this rule breaks IEEE Word export.
 
 6. DO NOT invent libraries, services, or features not present in the context.
 7. ANTI-REPETITION (Introduction only): When writing Introduction,
@@ -142,7 +143,7 @@ APPLICATION DOMAIN FOR [1] (Introduction, Methodology, System Architecture and I
 
 OUTPUT — IN THIS ORDER (labels help downstream parsing):
 PART 1 — PAPER BODY
-(Prose + ### headings + single Mermaid; [n] citations only; no paths/lines.)
+(Prose + ### headings only; [n] citations only; no paths/lines; NO Mermaid or diagram markup.)
 
 PART 2 — REFERENCES
 (Short IEEE-like list; [1] MUST use the repository URL when provided below; do not invent URLs.
diff --git a/export/ieee_template_export.py b/export/ieee_template_export.py
index 9a915fd..60618ef 100644
--- a/export/ieee_template_export.py
+++ b/export/ieee_template_export.py
@@ -135,6 +135,45 @@ def _line_starts_mermaid(stripped: str) -> bool:
     return any(stripped.startswith(k) for k in _MERMAID_KEYWORDS)
 
 
+# Writer artik Mermaid uretmemeli; kalinti satirlari Word'e dusmesin (PNG diyagram ayri).
+_MERMAID_BRACKET_EDGE_RE = re.compile(r"[A-Za-z0-9_]+\s*\[[^\]]*\]\s*-+[\>]?")
+
+
+def _is_mermaid_syntax_line(stripped: str) -> bool:
+    """Fence olmadan kalan Mermaid kenar/node satirlari (or. A[x] --> B[y])."""
+    if not stripped:
+        return False
+    s = stripped.strip()
+    if s.lower().startswith("```mermaid"):
+        return True
+    if _line_starts_mermaid(s):
+        return True
+    if _MERMAID_BRACKET_EDGE_RE.search(s):
+        return True
+    # Minimal kenar: NodeA --> NodeB (koseli etiket yok)
+    if re.search(r"\b[A-Za-z0-9_]+\s*-->\s*[A-Za-z0-9_]+\b", s):
+        return True
+    if re.match(r"^(subgraph\s|end\s*$|direction\s+[A-Z]{2})", s, re.IGNORECASE):
+        return True
+    return False
+
+
+def _is_mermaid_or_edge_code_blob(blob: str) -> bool:
+    """Acik kod blogunda Mermaid veya akis kenarlari varsa True (Word'e basilmaz)."""
+    t = (blob or "").strip()
+    if not t:
+        return False
+    lines = t.splitlines()
+    meaningful = [ln.strip() for ln in lines if ln.strip()]
+    first = meaningful[0] if meaningful else ""
+    if _line_starts_mermaid(first):
+        return True
+    if first.lower().startswith("```mermaid"):
+        return True
+    hits = sum(1 for ln in meaningful if _is_mermaid_syntax_line(ln))
+    return hits >= max(1, len(meaningful) // 2) or (len(meaningful) <= 4 and hits > 0)
+
+
 def _w_paragraph_style_id(p_el) -> str:
     """w:p elementinin w:pStyle/w:val degerini kucuk harfle dondurur; yoksa bos string."""
     if p_el is None:
@@ -629,8 +668,13 @@ def write_markdown_with_ieee_styles(
     def flush_code() -> None:
         if not code_lines:
             return
+        blob = "\n".join(code_lines)
+        # Writer'dan gelen Mermaid kalintisi ``` kod blogunda bile Word'e yazma.
+        if _is_mermaid_or_edge_code_blob(blob):
+            code_lines.clear()
+            return
         p = doc.add_paragraph(style=body_style)
-        _mono_runs(p, "\n".join(code_lines))
+        _mono_runs(p, blob)
         p.paragraph_format.left_indent = Pt(12)
         p.paragraph_format.space_after = Pt(6)
         code_lines.clear()
@@ -825,6 +869,9 @@ def write_markdown_with_ieee_styles(
         if last_h1_text and _stripped_norm == last_h1_text:
             continue
 
+        if _is_mermaid_syntax_line(stripped):
+            continue
+
         p = doc.add_paragraph(line, style=body_style)
         p.alignment = WD_PARAGRAPH_ALIGNMENT.JUSTIFY
         for r in p.runs:
````
