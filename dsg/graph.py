"""Deck State Graph (DSG) — the executable slide state at the core of SlideForge.

A presentation deck is both a visual artifact and an editable structured
document. The DSG is a directed graph G = (V, E) that bridges the two views:

* **Slide nodes** hold the rendered slide, its size, and background state
  (the cleaned background produced by perception-aligned decomposition).
* **Component nodes** hold the human-referable visual components recovered
  by Stage I (SAM3 segmentation + cross-modal VLM grounding): mask/bbox
  crops, taxonomy label (``text_type``), granularity (fine vs. merged),
  z-index, and text/LaTeX flags.
* **Object nodes** (optional) hold native PPTX objects when the source
  ``.pptx`` is available (extracted by ``decomposition/eval/decompose.py``),
  so components can be *bound* to directly editable structure.

Edges encode containment, reading order, z-order, merge-group membership,
and edit bindings between rendered components and native PPTX objects.

The graph is constructed from a Stage I run directory
(``decomposition/run_pipeline.py --run_dir ...``), whose per-slide
``final/metadata.json`` is the serialized component state:

    dsg = DeckStateGraph.from_decomposition_run("runs/my_deck")
    dsg.summary()
    dsg.to_json("dsg.json")

Downstream stages consume this state: the slide-native editing harness
grounds instructions to DSG nodes, and theme-preserving reconstruction
(``restyle/``) rebuilds each component node under a target aesthetic
contract while preserving the graph's layout and content
(``restyle/step0_ingest_dsg.py`` flattens the DSG into its per-page
workspace).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

@dataclass
class ComponentNode:
    """A human-referable visual component recovered by Stage I."""

    node_id: str
    slide_id: str
    filename: str                    # crop image under final/components/
    bbox_xyxy: list[float]
    text_type: str | None = None     # taxonomy label (306-class vocabulary)
    granularity: str | None = None   # "fine" | "merged"
    source: str | None = None        # which agent produced it
    score: float | None = None       # detector confidence
    z_index: int | None = None
    is_text_only: bool | None = None
    ocr_latex: str | None = None     # LaTeX transcription for formula crops
    polygon_xyxy: list | None = None
    segmentation_mode: str | None = None  # "mask" | "bbox"

    @property
    def area(self) -> float:
        x0, y0, x1, y1 = self.bbox_xyxy
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)


@dataclass
class ObjectNode:
    """A native PPTX object (only present when the source .pptx is known)."""

    node_id: str
    slide_id: str
    shape_type: str
    bbox_xyxy: list[float]
    text: str | None = None


@dataclass
class SlideNode:
    """One slide: rendered state + background + its component nodes."""

    slide_id: str
    image: str | None = None          # original render path
    width: int | None = None
    height: int | None = None
    cleaned_background: str | None = None  # background with foreground removed
    components: list[ComponentNode] = field(default_factory=list)
    objects: list[ObjectNode] = field(default_factory=list)


@dataclass
class Edge:
    """Directed relation between two DSG nodes.

    Kinds: ``contains`` (spatial containment), ``reads_before`` (reading
    order), ``above`` (z-order), ``merged_with`` (perceptual merge group),
    ``binds`` (component <-> native PPTX object edit binding).
    """

    src: str
    dst: str
    kind: str
    weight: float | None = None


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

_CONTAIN_FRAC = 0.9   # fraction of child area inside parent to add `contains`
_BIND_IOU = 0.5       # min IoU to bind a component to a native PPTX object


def _inter(a: list[float], b: list[float]) -> float:
    x0 = max(a[0], b[0]); y0 = max(a[1], b[1])
    x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


class DeckStateGraph:
    """The Deck State Graph: executable state for controllable slide editing."""

    def __init__(self) -> None:
        self.slides: list[SlideNode] = []
        self.edges: list[Edge] = []

    # -- construction --------------------------------------------------------

    @classmethod
    def from_decomposition_run(cls, run_dir: str | Path) -> "DeckStateGraph":
        """Build the DSG from a Stage I run directory.

        ``run_dir`` is the ``--run_dir`` given to
        ``decomposition/run_pipeline.py``; every child dir with a
        ``final/metadata.json`` becomes a slide node.
        """
        run_dir = Path(run_dir)
        g = cls()
        slide_dirs = sorted(
            d for d in run_dir.iterdir()
            if d.is_dir() and (d / "final" / "metadata.json").exists()
        )
        if not slide_dirs:
            raise FileNotFoundError(
                f"no slide dirs with final/metadata.json under {run_dir}")
        for sdir in slide_dirs:
            meta = json.loads(
                (sdir / "final" / "metadata.json").read_text(encoding="utf-8"))
            slide = SlideNode(slide_id=sdir.name, image=meta.get("image"))
            cleaned = sorted(sdir.glob("iter_*/segmentation/*/image_cleaned.png"))
            if cleaned:
                slide.cleaned_background = str(cleaned[0])
            comps = meta.get("final_components") or meta.get("components") or []
            for i, c in enumerate(comps):
                slide.components.append(ComponentNode(
                    node_id=f"{sdir.name}/c{i:03d}",
                    slide_id=sdir.name,
                    filename=c.get("filename", ""),
                    bbox_xyxy=[float(v) for v in c.get("bbox_xyxy", [0, 0, 0, 0])],
                    text_type=c.get("text_type"),
                    granularity=c.get("granularity"),
                    source=c.get("source"),
                    score=c.get("score"),
                    z_index=c.get("z_index"),
                    is_text_only=c.get("is_text_only"),
                    ocr_latex=c.get("ocr_latex"),
                    polygon_xyxy=c.get("polygon_xyxy"),
                    segmentation_mode=c.get("segmentation_mode"),
                ))
            g.slides.append(slide)
            g._derive_edges(slide)
        return g

    def _derive_edges(self, slide: SlideNode) -> None:
        comps = slide.components
        # reading order: top-to-bottom, then left-to-right
        order = sorted(comps, key=lambda c: (c.bbox_xyxy[1], c.bbox_xyxy[0]))
        for a, b in zip(order, order[1:]):
            self.edges.append(Edge(a.node_id, b.node_id, "reads_before"))
        for a in comps:
            for b in comps:
                if a is b:
                    continue
                # containment: most of b lies inside a, and a is larger
                if (b.area > 0 and a.area > b.area
                        and _inter(a.bbox_xyxy, b.bbox_xyxy) / b.area
                        >= _CONTAIN_FRAC):
                    self.edges.append(Edge(a.node_id, b.node_id, "contains"))
                # z-order among overlapping components
                if (a.z_index is not None and b.z_index is not None
                        and a.z_index > b.z_index
                        and _inter(a.bbox_xyxy, b.bbox_xyxy) > 0):
                    self.edges.append(Edge(a.node_id, b.node_id, "above"))

    # -- native PPTX binding -------------------------------------------------

    def bind_pptx_objects(self, slide_id: str,
                          pptx_components_json: str | Path) -> int:
        """Attach native PPTX object nodes (from
        ``decomposition/eval/decompose.py`` output) to a slide and create
        ``binds`` edges to the visually recovered components by IoU.
        Returns the number of bindings created."""
        slide = next(s for s in self.slides if s.slide_id == slide_id)
        shapes = json.loads(Path(pptx_components_json).read_text(encoding="utf-8"))
        if isinstance(shapes, dict):
            shapes = shapes.get("components", [])
        n_bound = 0
        for j, sh in enumerate(shapes):
            bbox = [float(v) for v in
                    (sh.get("bbox_xyxy") or sh.get("bbox") or [0, 0, 0, 0])]
            obj = ObjectNode(
                node_id=f"{slide_id}/o{j:03d}", slide_id=slide_id,
                shape_type=str(sh.get("shape_type", sh.get("type", "shape"))),
                bbox_xyxy=bbox, text=sh.get("text"))
            slide.objects.append(obj)
            for comp in slide.components:
                inter = _inter(bbox, comp.bbox_xyxy)
                union = (comp.area
                         + max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])
                         - inter)
                iou = inter / union if union > 0 else 0.0
                if iou >= _BIND_IOU:
                    self.edges.append(
                        Edge(comp.node_id, obj.node_id, "binds", round(iou, 3)))
                    n_bound += 1
        return n_bound

    # -- queries -------------------------------------------------------------

    def slide(self, slide_id: str) -> SlideNode:
        return next(s for s in self.slides if s.slide_id == slide_id)

    def components(self) -> list[ComponentNode]:
        return [c for s in self.slides for c in s.components]

    def summary(self) -> str:
        lines = [f"DeckStateGraph: {len(self.slides)} slide(s), "
                 f"{len(self.components())} component(s), "
                 f"{len(self.edges)} edge(s)"]
        for s in self.slides:
            kinds: dict[str, int] = {}
            for e in self.edges:
                if e.src.startswith(s.slide_id + "/"):
                    kinds[e.kind] = kinds.get(e.kind, 0) + 1
            kind_s = ", ".join(f"{k}={v}" for k, v in sorted(kinds.items()))
            lines.append(f"  {s.slide_id}: {len(s.components)} components"
                         f" ({kind_s})")
        return "\n".join(lines)

    # -- (de)serialization ---------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "slides": [asdict(s) for s in self.slides],
            "edges": [asdict(e) for e in self.edges],
        }

    def to_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
                        encoding="utf-8")
        return path

    @classmethod
    def from_json(cls, path: str | Path) -> "DeckStateGraph":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        g = cls()
        for s in d.get("slides", []):
            comps = [ComponentNode(**c) for c in s.pop("components", [])]
            objs = [ObjectNode(**o) for o in s.pop("objects", [])]
            g.slides.append(SlideNode(components=comps, objects=objs, **s))
        g.edges = [Edge(**e) for e in d.get("edges", [])]
        return g
