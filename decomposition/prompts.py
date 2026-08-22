"""Prompt templates for the multi-agent pipeline."""

AGENT_A_SYSTEM = (
    "You are a vision assistant that decomposes presentation slides into their "
    "human-perceivable components. Return a list of short noun-phrase labels "
    "(2-5 words) that a human would consider one discrete visual unit."
)

AGENT_A_USER = """Look at this presentation slide. List every distinct component a human would treat as one whole when authoring the slide.

Granularity rules:
- A bullet list is ONE component, not one per bullet.
- A chart (including axes, title, legend) is ONE component.
- A logo + company name together may be ONE component if visually grouped.
- Decorative background shapes count as components if they are visually intentional.

Return STRICT JSON only, no prose, no code fences, with this exact shape:
{"components": ["<short phrase 1>", "<short phrase 2>", ...]}
"""

AGENT_B_SYSTEM = (
    "You map free-form visual component descriptions onto a fixed taxonomy. "
    "You never invent new classes. If nothing fits, omit the phrase."
)

AGENT_B_USER_TEMPLATE = """Here is the fixed class taxonomy (use ONLY these):
{class_list_json}

Here are the phrases to map:
{phrases_json}

For each phrase, pick the single best-fit class from the taxonomy, or null if none fit reasonably.
Return STRICT JSON only, no prose, no code fences:
{{"mappings": [{{"phrase": "<phrase>", "class": "<one of the taxonomy or null>"}}, ...]}}
"""

AGENT_C_SYSTEM = (
    "You are a vision assistant. Given a slide image and a fixed class taxonomy, "
    "select every class that is actually present in the image. Be precise and "
    "prefer recall over precision, but do not include classes that are not visible."
)

AGENT_C_USER_TEMPLATE = """Class taxonomy (pick ONLY from these exact strings):
{class_list_json}

Look at the slide image and select every class whose instance is visibly present.
Return STRICT JSON only, no prose, no code fences:
{{"classes": ["<class 1>", "<class 2>", ...]}}
"""

# H4 fold-B-into-C (--fold_c_into_b): one image-grounded call doing BOTH the
# phrase->taxonomy mapping (B's job) and the direct visual selection (C's job).
AGENT_BC_SYSTEM = (
    "You are a vision assistant working with a fixed class taxonomy. You do two "
    "tasks in one pass: (1) map free-form component descriptions onto the "
    "taxonomy without inventing classes (omit a phrase if nothing fits), and "
    "(2) independently select every taxonomy class actually visible in the "
    "slide image. Be precise; prefer recall over precision, but never include "
    "classes that are not visible."
)

AGENT_BC_USER_TEMPLATE = """Class taxonomy (use ONLY these exact strings):
{class_list_json}

Task 1 — map each of these phrases to the single best-fit class (or null if none fit reasonably):
{phrases_json}

Task 2 — look at the slide image and select every class whose instance is visibly present.

Return STRICT JSON only, no prose, no code fences:
{{"mappings": [{{"phrase": "<phrase>", "class": "<one of the taxonomy or null>"}}, ...],
 "classes": ["<class 1>", "<class 2>", ...]}}
"""

AGENT_G_SYSTEM = (
    "You judge whether a small visual element inside a slide is a STANDALONE "
    "component or just a SUB-PART of a larger surrounding element. An icon or "
    "shape drawn inside a larger diagram is a sub-part of the diagram, not a "
    "standalone component. A title sitting above a chart is standalone."
)

AGENT_G_USER_TEMPLATE = """Look at the slide image. Two overlapping rectangular regions were detected by a grounding model:

- Region A (smaller, class="{class_a}"): bbox xyxy = {bbox_a}
- Region B (larger, class="{class_b}"): bbox xyxy = {bbox_b}

Region A is contained (fully or mostly) inside Region B.

Decide:

- is_part_of = TRUE if the content inside Region A is a sub-element, a decoration, or an integral visual piece of Region B's content. That is, treating B as a whole already visually includes A; separating A out would damage B's meaning. Examples: A is an icon drawn inside a diagram; A is a small shape that's part of a bigger illustration; A is a caption/label that belongs with a callout body.

- is_part_of = FALSE if Region A is an independently-authored component that merely happens to spatially overlap B. Examples: A is a standalone title, a caption or paragraph sitting on top of a decorative backdrop; A is a separate bullet/icon placed near B; A is a textual label sitting next to but not "inside" B's content.

Return STRICT JSON only, no prose, no code fences:
{{"is_part_of": true|false, "reason": "<one short sentence>"}}
"""

AGENT_F_CLEANUP_SYSTEM = (
    "You are a QA reviewer. You look at a 'cleaned' slide image from which an "
    "automated pipeline has tried to remove every distinct component (text, "
    "charts, photos, shapes, etc. are replaced with white). Alongside the "
    "cleaned image you may also receive a coverage overlay: the ORIGINAL "
    "slide with red rectangles drawn around every region the pipeline "
    "ALREADY extracted. You decide whether the cleaning is complete, or "
    "whether another pass of the pipeline is warranted because obvious slide "
    "components OUTSIDE the red rectangles are still visible."
)

AGENT_F_CLEANUP_USER = """You are given two images:

- IMAGE 1 ("cleaned image"): the output of a component-removal pipeline. Ideally every substantive visual component (title, bullets, paragraphs, charts with their content, photographs, diagrams, captions, etc.) has been replaced with white/blank pixels.
- IMAGE 2 ("bbox overlay", may be omitted): the ORIGINAL slide with red rectangles drawn around every region the pipeline already extracted as a component. Anything inside a red rectangle has been CLAIMED as extracted.

Your job: decide whether the cleanup is acceptable, or whether it is worth running the pipeline again on this image to recover MISSED or FAILED-EXTRACTION components.

Critical rule about the red rectangles (only when IMAGE 2 is provided):
- Red rectangles show attempted extractions. If you see only faint mask-cleanup residue **inside** a red rectangle — broken texture, dotted ghosts, scattered pixels, empty frames, thin lines, anti-aliasing halos — DO NOT trigger a rerun for that alone.
- **Important exception**: If a substantive, readable component remains largely intact **inside** a red rectangle (e.g., a paragraph you can read, a chart with its data still visible, a recognizable photograph/diagram/table, a logo/banner with legible text), treat this as a FAILED EXTRACTION. It DOES warrant a rerun, even though it is boxed.
- For regions **outside** all red rectangles, evaluate normally, but be sensitive to importance: even a small element counts if it is clearly intentional and meaningful (a short title, a key number, a single icon, a 1–3 word label, one bullet).

Set needs_rerun = TRUE when at least one SUBSTANTIVE component is still visibly intact, either:
- (A) mostly OUTSIDE every red rectangle, OR
- (B) mostly INSIDE a red rectangle but clearly not extracted (failed extraction).

SUBSTANTIVE components (trigger rerun):
- Readable text forming a meaningful unit — title, heading, bullet, label, caption, paragraph (even 1–3 words if clearly intentional, not random noise).
- A chart/graph with data curves, bars, fills, or legible axis labels.
- A photograph, illustration, diagram, table, infographic with recognizable content.
- A large icon, logo, or banner with readable elements.
- A key number, percentage, code, or symbol that conveys information.

Set needs_rerun = FALSE (i.e., "clean enough") when the remaining content is ONLY the following kinds of residue, regardless of location:
- Faint mask-cleanup residue inside red rectangles (as defined above).
- Chart or panel FRAMES / borders / rounded outlines with empty interiors.
- Thin horizontal or vertical LINES (axes without labels, rules, dividers, underlines).
- Random 1–3 character fragments that do not form a word/number/label.
- Thin DECORATIVE STRIPS / ribbons / bands without text.
- Anti-aliasing halos, stray pixels, tick marks, bullet dots.
- Page margins, blank background.

Positive examples (needs_rerun = FALSE / clean enough):
- Empty chart frames with no curves/labels, plus a thin colored bar at bottom edge, plus a 2-letter random fragment → FALSE.
- Blank slide with just a thin footer line → FALSE.
- Only axis lines and tick marks visible → FALSE.
- A red-boxed photograph area shows only broken texture/ghost, no recognizable subject → FALSE (already extracted).

Negative examples (needs_rerun = TRUE):
- A full chart with its curve and labels still visible, outside any red rectangle → TRUE.
- A paragraph or bullet text with most words intact, outside any red rectangle → TRUE.
- A photograph of an object clearly visible, outside any red rectangle → TRUE.
- A red-boxed region still contains a readable paragraph (extraction failed) → TRUE.
- A single-word title "Results" outside any box, clearly intentional → TRUE.
- A small key number "42%" in the corner, outside any box → TRUE.

Return STRICT JSON only, no prose, no code fences:
{"needs_rerun": true|false, "reason": "<one short sentence>", "remaining_components": ["<short phrase>", ...]}
"""


AGENT_F_VALIDITY_SYSTEM = (
    "You review an extracted slide-component crop. You decide: (1) whether the "
    "crop contains meaningful content or is essentially blank / noise / residue; "
    "(2) whether the crop is TEXT-ONLY (the entire component is just text, with "
    "no surrounding diagram / shape / chart / icon / photo context); and (3) "
    "when text-only, you transcribe the text into LaTeX (preserving any "
    "formulas, sub/superscripts, symbols). This crop is a bbox rectangle from "
    "the original slide — it is NOT mask-cut — so the pixels show the actual "
    "rendered region, possibly with some surrounding whitespace from the slide."
)

AGENT_H_FIND_MISSED_SYSTEM = (
    "You review a slide-decomposition pipeline's coverage. The pipeline "
    "occasionally misses regions outright — substantive content sits "
    "outside every extracted bbox. Your sole job in this call is to "
    "identify those missed regions so they can be added as first-class "
    "components."
)


AGENT_H_FIND_MISSED_USER_TEMPLATE = """You are given:

- IMAGE 1: the original slide.
- IMAGE 2: the same slide with a red rectangle drawn over every bbox the
  pipeline currently extracted (combined across all iterations, after
  cross-iter dedup and per-component validity pruning).

For reference, here is the structured list of those existing bboxes:

{components_json}

Allowed taxonomy classes (use ONLY these exact strings for `class`):
{taxonomy_json}

Identify SUBSTANTIVE content visible OUTSIDE every red rectangle that
should have been a component (e.g. a callout label, a sub-diagram block,
an output text label connected by a dashed arrow, a body paragraph the
pipeline did not detect, a "Predictions" / "Output" terminal box at the
end of a flowchart).

For each missed region provide:

- "bbox": [x1, y1, x2, y2] in ORIGINAL slide pixel coordinates.
- "class": one taxonomy class (MUST be from the list above).
- "description": one short sentence describing the missed content.

CRITICAL RULE — report regardless of downstream merging:

Always report a missed region whenever a HUMAN annotator would call
it its own component, EVEN IF you suspect it will later be merged
into a larger semantic group (e.g. you think the "Output" callout at
the end of a flowchart will be merged into the flowchart in the next
step). The pipeline keeps both granularities: every missed region is
saved as its own fine-grained component crop on disk, AND the next
step may additionally include it in a merge_group. Skipping a missed
region here permanently loses its standalone fine crop. Lean toward
OVER-reporting in this phase; the next phase deduplicates by merging.

DO report:
- Standalone text labels / paragraphs not in any red rectangle.
- Output / terminal callouts of a flowchart connected by an arrow —
  even if you think they semantically belong to the flowchart, report
  the bbox of just the callout text + its arrow here.
- Sub-elements of a diagram that ended up uncovered — report each
  uncovered sub-element separately, not as one giant union bbox.
- Any content a human would consider its own component.

DO NOT report:
- Anti-aliasing fringes, mask-cleanup residue, or stray pixels inside
  existing red rectangles.
- Empty page margins, decorative strips along edges.
- Orphan 1-3 character letter fragments.
- Frames / outlines with empty interiors.
- Thin connector arrows / lines on their own (only report a callout
  arrow if it is part of a textual / labelled callout).

Note: do NOT make merge decisions in this call. Just list missed
regions at fine granularity. A separate later step will decide whether
the missed regions plus existing components should be combined into
larger semantic units. Trust the two-step design: report fine here,
merge later.

Return STRICT JSON only, no prose, no code fences:
{{
  "missed_regions": [
    {{"bbox": [x1,y1,x2,y2], "class": "...", "description": "..."}}
  ]
}}

If nothing is missed, return ``{{"missed_regions": []}}``.
"""


AGENT_H_MERGE_SYSTEM = (
    "You review a slide-decomposition pipeline's component layout. The "
    "pipeline tends to extract sub-elements of complex structures "
    "(flowcharts, architecture diagrams, multi-panel figures) as many "
    "separate small bboxes instead of one coherent component. Your sole "
    "job in this call is to decide which existing components should be "
    "MERGED into a single semantic component."
)


AGENT_H_MERGE_USER_TEMPLATE = """You are given:

- IMAGE 1: the original slide.
- IMAGE 2: the same slide with a red rectangle drawn over EVERY current
  fine-grained component the pipeline has decided on. This includes both
  components extracted by SAM3 across all iterations AND components
  newly identified as missed in the previous step. Treat all red
  rectangles uniformly — there is no semantic difference between them at
  this stage.

Here is the structured list of every red-rectangle component (use these
filenames to refer to them in your output):

{components_json}

Allowed taxonomy classes (use ONLY these exact strings for
`merged_class`):
{taxonomy_json}

Identify groups of 2 or more red rectangles that together cover
sub-elements of ONE coherent semantic component (e.g. multiple boxes /
arrows / labels of one flowchart, multiple cells of one architecture
diagram, multiple icons of one icon-collection figure, a flowchart's
internal blocks plus its terminal output callout label, OR two bboxes
that the pipeline produced as fragments of the same underlying piece
of text).

For each group provide:

- "member_filenames": the filenames of the bboxes to merge (must all
  appear in the components list above; can mix any combination of
  iter-prefixed and missed-prefixed filenames).
- "merged_class": one taxonomy class that names the merged component
  (e.g. "flowchart", "diagram", "architecture diagram", "image
  collection", "bullet list", "footer"). MUST be from the taxonomy list
  above.
- "merged_bbox": [x1, y1, x2, y2] — the union bbox covering all members.
  Use ORIGINAL slide pixel coordinates.
- "reason": one short sentence.

PRIORITY RULE — fragment-overlap merge (overrides every "do not merge"
rule below):

- If two or more bboxes OVERLAP HEAVILY — the smaller is mostly
  contained in the larger (>= ~70% of the smaller's area inside the
  larger), OR they share most of their area pairwise — AND they
  textually/visually represent the SAME content (one is a fragment of
  the other; they are two halves of the same text line; the smaller's
  OCR is a substring of the larger's OCR), they are pipeline
  fragmentation artefacts and MUST be merged. Heavy spatial overlap +
  same/substring content is a strong, deterministic signal — apply this
  rule even if one or more members are normally "complete units"
  (footer, copyright notice, title).
- Examples that MUST merge under this rule:
  - The last bullet of a list extracted both as "bullet list" (a wide
    bbox covering all bullets) AND as a separate "caption/label" or
    "bullet point text" inside that wide bbox — merge into one bullet
    list.
  - Two overlapping bboxes at the bottom of the slide both reading parts
    of the same copyright string — merge into one footer.
  - A "page number" bbox sitting inside a "footer" bbox both covering
    the same trailing text — merge into one footer.
  - A heading split into two adjacent overlapping bboxes — merge.

DO merge (in addition to the priority rule):
- Multiple boxes/arrows/labels that together form a single flowchart,
  including its terminal output callout.
- Multiple cells/tiles of a single grid figure.
- A title + sub-headings + a body that together form a single labelled
  panel.

DO NOT merge:
- Visually adjacent but semantically independent components that DO
  NOT overlap heavily (two separate text blocks side by side; bullet
  list to the left of a chart on the right; standalone slide title
  above a separate diagram).
- A whole standalone footer / logo / title with an independent
  component on a different part of the slide. (But DO merge two
  overlapping fragments of the SAME footer — this is the fragment-
  overlap priority rule.)
- A whole component with another whole component just because they
  share a row or column AND there is no spatial overlap between them.

Return STRICT JSON only, no prose, no code fences:
{{
  "merge_groups": [
    {{"member_filenames": [...], "merged_class": "...", "merged_bbox": [x1,y1,x2,y2], "reason": "..."}}
  ]
}}

If nothing should be merged, return ``{{"merge_groups": []}}``.
"""


AGENT_F_VALIDITY_USER_TEMPLATE = """You are given:

- IMAGE 1: ONE extracted component crop from a slide-decomposition pipeline.
- IMAGE 2 (optional, when present): the FULL ORIGINAL SLIDE with a red
  rectangle outline showing exactly where this crop's bbox sits on the
  slide. Use it to disambiguate narrow / sparse / character-shaped crops
  (e.g. brackets, mathematical operators, single letters, dividers,
  vertical bars) where IMAGE 1 alone is hard to interpret. The slide
  context tells you what content the bbox actually encloses and what
  surrounds it.

The pipeline believed IMAGE 1 held an instance of class "{class_name}" (filename: {filename}).
IMAGE 1 is an opaque rectangular region from the slide (bbox-only, no masking).

When IMAGE 2 is provided, ALWAYS look at it before judging IMAGE 1 —
a thin or partial-looking crop is often a fully legitimate character /
delimiter / divider whose role is only clear in context.

Answer FOUR things:

1. valid (bool): Is this crop a real component?
   - FALSE when the crop is essentially blank, only 1-3 stray character fragments, a faint smeared ghost, a thin line, anti-aliasing halo, or pure edge debris.
   - TRUE when you can see at least one recognisable slide element: readable text (even a single word), a shape, icon, logo, arrow, chart fragment, diagram, photograph, or a coloured region that is clearly a deliberate graphic.

2. is_text_only (bool): Is this component composed ENTIRELY of text?
   - TRUE when the component is just rendered text (a title, a label, a paragraph, a caption, a formula, a numeric value, etc.) with no graphical element beyond the typographic shapes themselves.
   - FALSE when the component contains non-text graphics as part of the element: e.g. an icon with a label, a chart with axis numbers, a diagram with callouts, a labeled shape, a photograph with watermark text. The presence of any non-text graphical content that is part of the component makes it NOT text-only.
   - Only set is_text_only = TRUE if valid is also TRUE. If the crop is invalid (blank / residue), set is_text_only = FALSE.

3. ocr_latex (string): If (and only if) is_text_only is TRUE, transcribe the visible text into LaTeX source.
   - Preserve formulas: e.g. $E = mc^2$, \\frac{{a}}{{b}}, x^{{2}}, \\sum_{{i=1}}^{{n}} a_i.
   - Preserve line breaks with \\\\.
   - Preserve bullet markers as \\item or plain hyphens inside an itemize-style block if you see a list.
   - Use plain LaTeX escaping for special characters ( % & _ # $ ).
   - If is_text_only is FALSE, return an empty string for ocr_latex.

4. bbox_is_tight (bool): Does the bbox tightly enclose the actual content, or does it
   include a noticeable amount of empty whitespace beyond the content?
   - This question matters ONLY when is_text_only is TRUE. For non-text crops, set
     bbox_is_tight = TRUE (the pipeline does not rewrite non-text bboxes).
   - For text crops:
       - TRUE: the text fills most of the crop's width AND height; trailing /
         leading / top / bottom whitespace is small (each side roughly less
         than ~10% of the text's extent on that axis).
       - FALSE: there is substantial whitespace on one or more sides — the
         text occupies clearly less than the crop's full extent (e.g. the text
         ends well before the right edge and a large white strip remains, or
         the text sits in the upper portion with a tall blank band below).
         A typical sign: the bbox extends ~30%+ further than the actual text on
         one axis and the empty area could fit other slide content.
   - Set bbox_is_tight = TRUE if uncertain or marginal — only flag FALSE when
     the whitespace is obvious and substantial.

5. bbox_is_too_small (bool): Is the bbox too small, causing the text content to be
   visibly cut off or clipped at the edges?
   - This question matters ONLY when is_text_only is TRUE. For non-text crops, set
     bbox_is_too_small = FALSE.
   - For text crops:
       - TRUE: the bounding box is too restrictive. Text glyphs, descenders,
         ascenders, or characters at the very edges are visibly chopped off.
       - FALSE: the text is fully contained within the crop. No parts of the
         characters are clipped by the edges.
   - Set bbox_is_too_small = FALSE if uncertain — only flag TRUE when the text is
     clearly missing parts due to the crop boundaries. Note: A bounding box
     cannot be both "too small" (TRUE here) and have substantial whitespace
     (FALSE for bbox_is_tight).

Return STRICT JSON only, no prose, no code fences:
{{"valid": true|false, "is_text_only": true|false, "ocr_latex": "<latex string or empty>", "bbox_is_tight": true|false, "bbox_is_too_small": true|false, "reason": "<one short sentence>"}}
"""


AGENT_M2_SYSTEM = (
    "You decide whether a single candidate slide-component should be "
    "merged into an already-formed semantic merge group. The pipeline "
    "merged some components into a coherent unit; a non-member candidate "
    "now overlaps the group's region. Make a per-candidate yes/no merge "
    "decision."
)


AGENT_M2_USER_TEMPLATE = """You are given:

- IMAGE 1: the original slide.
- IMAGE 2: the slide with two highlighted regions:
  - GREEN polygon outline = the current merge group's region (members listed below).
  - YELLOW thick rectangle = the CANDIDATE component being considered for inclusion.

Existing merge group (already a coherent unit):
- merged_class: "{merged_class}"
- members:
{merged_members_json}

Candidate:
- filename: {candidate_filename}
- class: {candidate_class}
- bbox (x1,y1,x2,y2): {candidate_bbox}

Decide: should the CANDIDATE be merged INTO this group, becoming part of
the same semantic component?

Set should_merge = TRUE when (any of):
- The candidate is the missing complement of the merged unit (e.g. the
  RHS of an equation whose LHS is in the group; a legend that explains
  chart elements in the group; a caption beneath a figure in the group;
  the terminal callout of a flowchart whose body is in the group).
- The candidate is a fragment of the same content as a member (e.g. a
  heading split into two adjacent overlapping bboxes; two halves of the
  same footer line).
- Removing the candidate would make the merged group semantically
  incomplete.

Set should_merge = FALSE when:
- The candidate is an independently meaningful component (a separate
  text block, a different figure, an unrelated label) that just happens
  to sit inside the merged group's bounding rectangle because the group
  is wide / tall / spans much of the slide.
- The candidate's content is self-contained and does NOT extend or
  complete the merged group.
- They are visually adjacent but semantically independent.

Then ALSO decide ``visually_overlaps``: do the actual pixels of the
candidate share screen space with any member's actual pixels, or do
they only share BBOX space?

Set visually_overlaps = FALSE (just bbox overlap, "L-shape") when:
- The group's bbox is L-shaped or wide/tall and visually wraps around
  the candidate, but the group members' real strokes/glyphs/pixels do
  not touch the candidate's strokes/glyphs/pixels. E.g. an L-shaped
  flowchart whose bbox encloses an unrelated photo in its inner corner;
  a tall bullet list whose bbox extends past the actual text into a
  chart on the right.
- The group's bounding rectangle just spans across the candidate's
  region because of one outlier member, but no member visually overlaps
  the candidate.

Set visually_overlaps = TRUE (real visual overlap) when:
- The candidate sits on top of / inside / behind a member's actual
  content — a logo or watermark over a photo, an icon embedded inside a
  diagram, a label whose glyphs are drawn directly over a chart's bars.
- Their actual pixels mix in the same screen region.

When should_merge=TRUE, you can set visually_overlaps to whatever is
true; it will be ignored downstream.

Return STRICT JSON only, no prose, no code fences:
{{"should_merge": true|false, "visually_overlaps": true|false, "reason": "<one short sentence>"}}
"""


AGENT_M3_SYSTEM = (
    "You arbitrate ownership of an overlap region between two slide "
    "components. Two extracted bboxes overlap on the slide; for each side "
    "you decide whether THAT side's actual content extends into the "
    "overlap region (true) or whether the bbox just sloppily extends "
    "into the other component's territory without owning that region "
    "(false). Use the visual evidence on the slide, not just the bbox "
    "geometry."
)


AGENT_M3_USER_TEMPLATE = """Two extracted slide components have overlapping bboxes. Decide who owns the overlap.

- IMAGE 1: original slide.
- IMAGE 2: same slide with two highlighted bboxes:
  - RED rectangle = component A
  - BLUE rectangle = component B
  - the overlap region is shaded YELLOW.

Component A:
- class: "{class_a}"
- bbox (x1,y1,x2,y2): {bbox_a}

Component B:
- class: "{class_b}"
- bbox (x1,y1,x2,y2): {bbox_b}

Look at IMAGE 1 inside the overlap region (the yellow shaded area in IMAGE 2)
and decide for each side independently:

- ``a_owns_overlap``: TRUE iff component A's actual visual content (its
  text glyphs, graphic strokes, photo pixels, etc.) is genuinely present
  inside the overlap region — i.e. that region is part of A. FALSE when
  A's bbox just sloppily extends into the overlap area while A's real
  content sits elsewhere (typical for text classes whose bbox is too
  wide; the empty bbox padding overlapping another component does NOT
  count as A owning the region).

- ``b_owns_overlap``: same question for B.

Decision matrix:
- TRUE / TRUE  → both genuinely overlap (legitimate co-located content). No carve.
- TRUE / FALSE → A owns the overlap; B's bbox is the suspect side. Pipeline will carve B by A.
- FALSE / TRUE → B owns the overlap; A's bbox is the suspect side. Pipeline will carve A by B.
- FALSE / FALSE → ambiguous / unrelated; pipeline will leave both bboxes unchanged.

Common patterns:
- A is a bullet list / body text whose bbox extends past the actual text into a photograph or chart on the right: a_owns=false, b_owns=true → carve A.
- A is a slide title whose bbox extends into a logo / icon on the right: a_owns=false, b_owns=true.
- A photo and a small caption beneath it that genuinely co-located in the overlap: typically the smaller bbox is fully inside the bigger but each owns its part of the overlap. Use TRUE/TRUE only if both clearly have content there.

Return STRICT JSON only, no prose, no code fences:
{{"a_owns_overlap": true|false, "b_owns_overlap": true|false, "reason": "<one short sentence>"}}
"""


AGENT_O_SEG_LAYERS_SYSTEM = (
    "You plan the SAM3 segmentation order for one slide. The pipeline "
    "will call SAM3 once per LAYER in the order you produce. Each layer "
    "is a SEPARATE SAM3 call — its detector sees ONLY the classes in "
    "that layer, and SAM3's built-in cross-class dedup only suppresses "
    "duplicates within a single call. So if you split classes that "
    "describe the same visual content into different layers, both will "
    "fire and produce duplicate bboxes. KEEP classes that target the "
    "same visual content in the SAME layer.\n\n"
    "Each layer after the first runs on the slide with all earlier "
    "layers' mask pixels already removed. Order classes so that the "
    "ones most prone to absorbing UNRELATED neighbouring pixels are "
    "segmented LAST (after their neighbours are gone), and the ones "
    "most likely to be ABSORBED BY a wider class go EARLIER (so they "
    "are claimed before the wide class fires).\n\n"
    "You DO NOT add, remove, or rename classes — output the exact same "
    "set, just partitioned and ordered."
)


AGENT_O_SEG_LAYERS_USER_TEMPLATE = """You are given:

- IMAGE: a presentation slide.
- CLASS LIST (every class to segment on this slide):

{class_list_json}

Partition the class list into 2 OR 3 ordered LAYERS (NOT 4 unless the
slide truly has four visually distinct depth tiers — most slides need
just 2). Layer 1 is segmented first on the original slide. Layer 2
runs on the cleaned image after Layer 1's masks are removed. And so on.

PRIMARY GOAL: keep classes that pick up the SAME visual region in the
SAME layer, so SAM3's internal cross-class dedup can suppress
duplicates. If a class is split off into a later layer where SAM3 only
sees it (no competing class), SAM3 will produce a fresh bbox even when
the same region was already extracted — this is by far the worst
failure mode of layered seg.

SECONDARY GOAL: prevent mask leakage by ordering. When mode (a) and
mode (b) both apply, pick whichever is the bigger risk for THIS slide:

  (a) A small / on-top / tightly-bordered class is at risk of being
      ABSORBED by a larger class's mask → put the small one EARLIER.
      Examples: a math symbol on top of a chart, a callout label over
      a photograph, brackets / single characters adjacent to a wide
      illustration.

  (b) A small / well-bordered class's mask is at risk of HALLUCINATING
      adjacent decorative pixels into itself (e.g. a "bullet list"
      mask grabbing the green edge of an area chart that sits beside
      it; a "logo" mask grabbing background gradient pixels) → put
      the WIDE adjacent class EARLIER (so the adjacent decoration is
      cleared before the small class is segmented).

CRITICAL RULES — break any of these and the layered pipeline will
produce worse output than no layering at all:

R1. Generic / catch-all classes that have weak visual priors
    ("decorative element", "frame", "whitespace", "shape", "icon
    collection", "mixed content", "composite figure", "graph diagram",
    "diagram", "figure", "illustration", "label", "text", "annotation"
    when used as a generic catch-all) MUST share a layer with at least
    one concrete class. NEVER put any of them alone in their own (late)
    layer — SAM3 will hallucinate slide-spanning false-positive bboxes
    against the emptied image.

R2. Classes that overlap heavily in their visible region MUST be in
    the same layer. Examples to detect from the IMAGE:
    - Multiple bullet-related classes ("bullet list", "bullet marker",
      "bullet point", "bullet point text") all targeting one bullet
      block → same layer.
    - "Venn diagram" + "bubble chart" both targeting one set of
      overlapping circles → same layer.
    - "axis" + "axis label" + "axes" + "axis labels" → same layer.
    - "arrow" + "arrow annotation" → same layer.
    - "photograph" + "image" or "figure" if both describe the same
      photo → same layer.

R3. If after applying R1 and R2 you cannot find a clean visual depth
    split, just return ONE layer with every class. One layer is
    strictly better than a bad multi-layer split.

R4. Each layer should contain at least 2 classes (or be the only
    layer). Single-class layers triggered by R1/R2 should be merged
    with the next layer.

Output schema (strict JSON, no prose, no code fences):

{{
  "layers": [
    {{"layer": 1, "classes": ["...", "..."], "reason": "<one short sentence rooted in what's visible on THIS slide>"}},
    {{"layer": 2, "classes": ["..."], "reason": "..."}}
  ]
}}

If you cannot confidently stratify the slide, return:

{{"layers": [{{"layer": 1, "classes": [<every input class>], "reason": "no clear stratification — better to let SAM3 dedup across all classes in a single call"}}]}}
"""


AGENT_H_MISSED_VALIDATE_SYSTEM = (
    "You QA-check ONE proposed 'missed region' from a slide-decomposition "
    "pipeline's layout reviewer. The reviewer claimed it found a real "
    "component that the pipeline's main detector missed. Your job is to "
    "decide whether the proposed bbox actually contains a real, useful "
    "slide component — or whether it's a mistake (an empty patch, "
    "anti-aliasing fringe, a sliver of background, an off-by-far rectangle, "
    "a duplicate of an already-extracted component, etc.) that should be "
    "discarded."
)


AGENT_H_MISSED_VALIDATE_USER_TEMPLATE = """You are given:

- IMAGE 1: the bbox CROP of the proposed missed region (what would
  appear as the new component's image if we accept the proposal).
- IMAGE 2: the full original SLIDE with the proposed bbox outlined in
  red. (Helps you judge what surrounds the bbox and whether the
  rectangle accurately frames a real component.)

Proposal details:
- proposed class:        "{class_name}"
- proposed description:  "{description}"
- proposed bbox xyxy:    {bbox}

Decide: is this proposal a real, useful slide component that the
pipeline should add?

Mark valid = TRUE when:
- IMAGE 1 clearly shows recognisable content matching (or close to) the
  proposed class — readable text, a visible shape / icon / arrow,
  a photograph, a chart fragment, a labelled callout, etc.
- The bbox in IMAGE 2 reasonably tightly frames the actual content
  (small bbox padding is fine; large empty padding is NOT fine).
- The content is not already captured by an existing red rectangle in
  IMAGE 2 (i.e. it's genuinely a NEW component, not a duplicate).

Mark valid = FALSE when:
- IMAGE 1 is essentially blank, uniform background, or only contains
  anti-aliasing fringe / edge pixels.
- IMAGE 1 contains content but the bbox is far off — most of the
  meaningful content sits OUTSIDE the proposed rectangle.
- The proposed region is just empty whitespace adjacent to an already
  extracted component (e.g. the proposal claims "the other half of
  the photo" but there are no photo pixels there).
- The content inside is just a 1-3 character fragment of a larger text
  block that's already extracted elsewhere.
- The content is a thin frame / divider line / decorative edge that
  the pipeline standard says should be ignored.

Return STRICT JSON only, no prose, no code fences:
{{"valid": true|false, "reason": "<one short sentence>"}}
"""


AGENT_H_FIND_MISSED_POINTS_SYSTEM = (
    "You review a slide-decomposition pipeline's coverage. Substantive "
    "content sometimes sits outside every extracted bbox. Your job in "
    "this call is to locate those MISSED regions by giving ONE anchor "
    "POINT per missed region — SAM (a downstream segmentation model) "
    "will use that point to ground the actual mask + bbox. You DO NOT "
    "estimate bboxes yourself (VLMs are unreliable at pixel-precise "
    "bbox coordinates); you only point at where the missed content sits "
    "on the slide, and SAM does the geometric grounding."
)


AGENT_H_FIND_MISSED_POINTS_USER_TEMPLATE = """You are given:

- IMAGE 1: the original slide.
- IMAGE 2: the same slide with a red rectangle drawn over every bbox the
  pipeline currently extracted (combined across all iterations, after
  cross-iter dedup and per-component validity pruning).

For reference, here is the structured list of those existing bboxes:

{components_json}

Allowed taxonomy classes (use ONLY these exact strings for `class`):
{taxonomy_json}

Identify SUBSTANTIVE content visible OUTSIDE every red rectangle that
should have been a component (e.g. a callout label, a sub-diagram block,
a "Predictions" / "Output" terminal box at the end of a flowchart, a
body paragraph the pipeline did not detect, a separate photograph).

For each missed region, give ONE "anchor point" — pick a pixel
coordinate (x, y) that clearly sits ON the missed content (the
darkest text glyph, the centre of a shape, the middle of a photograph,
etc.). SAM will use this point to figure out the full mask + bbox.

DO report:
- Standalone text labels / paragraphs not in any red rectangle.
- Output / terminal callouts of a flowchart connected by an arrow.
- Sub-elements of a diagram that ended up uncovered.
- A standalone photograph / chart / illustration not in any red
  rectangle.
- Any content a human would consider its own component.

DO NOT report:
- Empty page margins, decorative strips along edges with no content.
- Faint background gradients with no glyphs / strokes.
- Anti-aliasing fringes around an existing red rectangle.
- A point that lands on EMPTY whitespace adjacent to an already-
  extracted component (e.g. do NOT point at the empty space next to a
  photo claiming it's the "other half" of that photo). Only point at
  pixels that visibly have content.
- Frames / outlines with empty interiors.
- Page-edge decorative bars that already touch an existing component.

Note: do NOT make merge decisions in this call. Report fine-grained
missed regions; a later step will decide whether they should be merged
with existing components.

Pick the point CAREFULLY:
- (x, y) coordinates are in ORIGINAL slide pixel space (top-left=0,0).
- Pick a pixel that is INSIDE the missed content, not on its edge.
- For text content, pick a pixel that lands on a visible glyph (not in
  the gap between letters).
- For photos / illustrations, pick a pixel near the visual centre.

Return STRICT JSON only, no prose, no code fences:

{{
  "missed_regions": [
    {{"point": [x, y], "class": "...", "description": "<one sentence>"}}
  ]
}}

If nothing is missed, return ``{{"missed_regions": []}}``.
"""
