"""Shared helpers: load metadata, resolve original slide image, build reconstructed canvas.

The "final" components in metadata.json already encode the merge-vs-fine decision:
they keep merged_xxx.png when a merge consumed fine components, and keep the
individual fine components when nothing merged them. So eval just iterates that list.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from PIL import Image


@dataclass
class FinalComponent:
    filename: str
    bbox_xyxy: tuple[float, float, float, float]
    text_type: str
    is_text_only: bool
    ocr_latex: str
    granularity: str
    polygon_xyxy: list[list[float]] | None
    merged_from: list[str] | None
    segmentation_mode: str = "bbox"

    @property
    def is_merged(self) -> bool:
        return self.granularity == "merged" or self.filename.startswith("merged_")

    @property
    def bbox_int(self) -> tuple[int, int, int, int]:
        x0, y0, x1, y1 = self.bbox_xyxy
        return int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))


@dataclass
class SlideEvalSample:
    slide_dir: str
    slide_id: str
    original_image_path: str
    final_dir: str
    components_dir: str
    final_components: list[FinalComponent]
    image_size: tuple[int, int]  # (W, H) of original slide
    reconstruction_override: str | None = None


def _resolve_original_image(metadata_image: str, slide_dir: str) -> str:
    """metadata.json stores the original image as an absolute path.
    Fall back to a few sibling locations if it's not found."""
    if os.path.exists(metadata_image):
        return metadata_image
    base = os.path.basename(metadata_image)
    candidates = [
        os.path.join(slide_dir, base),
        os.path.join(slide_dir, "original.png"),
        os.path.join(slide_dir, "iter_00", base),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Cannot locate original slide image for {slide_dir}; tried {[metadata_image, *candidates]}"
    )


def load_slide_sample(slide_dir: str) -> SlideEvalSample:
    slide_id = os.path.basename(os.path.normpath(slide_dir))
    
    # Standard SAM3 pipeline output format
    final_dir = os.path.join(slide_dir, "final")
    components_dir = os.path.join(final_dir, "components")
    meta_path = os.path.join(final_dir, "metadata.json")
    
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        original_image_path = _resolve_original_image(meta["image"], slide_dir)
        with Image.open(original_image_path) as im:
            W, H = im.size
        components = []
        for c in meta["final_components"]:
            components.append(
                FinalComponent(
                    filename=c["filename"],
                    bbox_xyxy=tuple(c["bbox_xyxy"]),
                    text_type=c.get("text_type", ""),
                    is_text_only=bool(c.get("is_text_only", False)),
                    ocr_latex=c.get("ocr_latex", "") or "",
                    granularity=c.get("granularity", "fine"),
                    polygon_xyxy=c.get("polygon_xyxy"),
                    merged_from=c.get("merged_from"),
                    segmentation_mode=c.get("segmentation_mode", "bbox"),
                )
            )
        return SlideEvalSample(
            slide_dir=slide_dir,
            slide_id=slide_id,
            original_image_path=original_image_path,
            final_dir=final_dir,
            components_dir=components_dir,
            final_components=components,
            image_size=(W, H),
        )

    # PPTX Baseline Format
    pptx_files = glob.glob(os.path.join(slide_dir, "*.pptx"))
    if pptx_files:
        pptx_path = pptx_files[0]
        base_name = os.path.splitext(os.path.basename(pptx_path))[0]
        pptx_json_path = os.path.join(slide_dir, f"{base_name}_components.json")
        reconstruction_path = os.path.join(slide_dir, f"{base_name}_reconstruction.png")
        
        # Auto-run decompose if results don't exist
        if not os.path.exists(pptx_json_path) or not os.path.exists(reconstruction_path):
            import sys
            decompose_mod_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            if decompose_mod_path not in sys.path:
                sys.path.insert(0, decompose_mod_path)
            from decompose import decompose_slide_single_rendering
            comp_dir = os.path.join(slide_dir, f"{base_name}_components")
            decompose_slide_single_rendering(pptx_path, comp_dir, pptx_json_path)
            
        with open(pptx_json_path) as f:
            meta = json.load(f)
            
        # Locate the Ground Truth image (any png that is not reconstruction or annotated)
        original_image_path = None
        for f in os.listdir(slide_dir):
            if f.endswith(".png") and not f.endswith("_reconstruction.png") and not f.endswith("_annotated.png"):
                original_image_path = os.path.join(slide_dir, f)
                break
                
        if not original_image_path:
            raise FileNotFoundError(f"Could not find original Ground Truth .png in {slide_dir}")
            
        with Image.open(original_image_path) as im:
            W, H = im.size
            
        w_pptx = meta["slide_info"]["dimensions_pixels"]["width"]
        h_pptx = meta["slide_info"]["dimensions_pixels"]["height"]
        
        scale_x = W / w_pptx if w_pptx > 0 else 1.0
        scale_y = H / h_pptx if h_pptx > 0 else 1.0
            
        components = []
        for c in meta["components"]:
            shape_type = c.get("type", "").lower()
            text_content = c.get("text", "")
            has_text = bool(text_content.strip())
            
            is_text_only = False
            text_type = ""
            if has_text:
                if "text_box" in shape_type or "textbox" in shape_type or "placeholder" in shape_type:
                    text_type = "paragraph"
                    is_text_only = True
                else:
                    text_type = "mixed"
                    
            components.append(
                FinalComponent(
                    filename=os.path.basename(c["image_path"]),
                    bbox_xyxy=(
                        float(c["position_pixels"]["left"]) * scale_x,
                        float(c["position_pixels"]["top"]) * scale_y,
                        float(c["position_pixels"]["right"]) * scale_x,
                        float(c["position_pixels"]["bottom"]) * scale_y
                    ),
                    text_type=text_type,
                    is_text_only=is_text_only,
                    ocr_latex=text_content,
                    granularity="fine",
                    polygon_xyxy=None,
                    merged_from=None,
                    segmentation_mode="bbox"
                )
            )
            
        return SlideEvalSample(
            slide_dir=slide_dir,
            slide_id=slide_id,
            original_image_path=original_image_path,
            final_dir=slide_dir,
            components_dir=os.path.join(slide_dir, f"{base_name}_components"),
            final_components=components,
            image_size=(W, H),
            reconstruction_override=reconstruction_path
        )
        
    raise FileNotFoundError(f"Could not find valid eval metadata or PPTX in {slide_dir}")

def iter_slide_dirs(root: str) -> Iterable[str]:
    """Yield each slide subdirectory under the cost_measure root."""
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if not os.path.isdir(p):
            continue
            
        # Format 1: SAM3 Pipeline
        if os.path.exists(os.path.join(p, "final", "metadata.json")):
            yield p
            continue
            
        # Format 2: PPTX Evaluation Format
        pptx_files = glob.glob(os.path.join(p, "*.pptx"))
        if pptx_files:
            yield p


def fill_alpha_holes(img: Image.Image) -> Image.Image:
    """Fill internal transparent holes inside the outer boundaries of an RGBA image."""
    if img.mode != "RGBA":
        return img
    try:
        import cv2
        r, g, b, alpha = img.split()
        alpha_bin = np.array(alpha)
        # Find outer contours of the mask (foreground is 255)
        contours, _ = cv2.findContours(alpha_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Draw filled contours on a blank mask to close all internal holes
        alpha_filled = np.zeros_like(alpha_bin)
        cv2.drawContours(alpha_filled, contours, -1, 255, -1)
        return Image.merge("RGBA", (r, g, b, Image.fromarray(alpha_filled)))
    except Exception:
        return img


def load_original_rgb(sample: SlideEvalSample) -> np.ndarray:
    with Image.open(sample.original_image_path) as im:
        return np.array(im.convert("RGB"))


def build_reconstruction(sample: SlideEvalSample, background: str = "white") -> np.ndarray:
    """Composite each final component crop onto a blank canvas at its bbox top-left.
    If reconstruction_override is set (e.g. for PPTX evaluation), return it directly.
    """
    if getattr(sample, "reconstruction_override", None) and os.path.exists(sample.reconstruction_override):
        with Image.open(sample.reconstruction_override) as im:
            im = im.convert("RGB")
            if im.size != sample.image_size:
                im = im.resize(sample.image_size, Image.LANCZOS)
            return np.array(im)

    W, H = sample.image_size
    if background == "white":
        canvas = np.full((H, W, 3), 255, dtype=np.uint8)
    elif background == "black":
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
    else:
        raise ValueError(f"Unknown background: {background}")

    import collections
    import heapq

    comps = sample.final_components
    N = len(comps)

    # Precompute area for each component to use for base sorting / tie-breaking
    areas = []
    for comp in comps:
        x0, y0, x1, y1 = comp.bbox_int
        areas.append((x1 - x0) * (y1 - y0))

    # Initialize graph structures for Kahn's topological sort.
    # An edge u -> v means u must be drawn BEFORE v (i.e. u is underneath v).
    adj = collections.defaultdict(list)
    in_degree = [0] * N

    # Pairwise check for overlap and determine relative layering constraints
    for i in range(N):
        x0_i, y0_i, x1_i, y1_i = comps[i].bbox_int
        for j in range(i + 1, N):
            x0_j, y0_j, x1_j, y1_j = comps[j].bbox_int
            
            # Calculate intersection box
            ox0 = max(x0_i, x0_j)
            oy0 = max(y0_i, y0_j)
            ox1 = min(x1_i, x1_j)
            oy1 = min(y1_i, y1_j)
            
            if ox1 > ox0 and oy1 > oy0:
                # Overlap exists! Load crops and compare white pixel ratio in the intersection.
                crop_path_i = os.path.join(sample.components_dir, comps[i].filename)
                crop_path_j = os.path.join(sample.components_dir, comps[j].filename)
                
                if os.path.exists(crop_path_i) and os.path.exists(crop_path_j):
                    try:
                        with Image.open(crop_path_i) as img_i, Image.open(crop_path_j) as img_j:
                            # Crop to local intersection
                            lx0_i, ly0_i = ox0 - x0_i, oy0 - y0_i
                            lx1_i, ly1_i = ox1 - x0_i, oy1 - y0_i
                            lx0_j, ly0_j = ox0 - x0_j, oy0 - y0_j
                            lx1_j, ly1_j = ox1 - x0_j, oy1 - y0_j
                            
                            slice_i = img_i.crop((lx0_i, ly0_i, lx1_i, ly1_i))
                            slice_j = img_j.crop((lx0_j, ly0_j, lx1_j, ly1_j))
                            
                            def get_white_ratio(sl: Image.Image) -> float:
                                arr = np.array(sl.convert("RGB"))
                                white_pixels = (arr[..., 0] >= 250) & (arr[..., 1] >= 250) & (arr[..., 2] >= 250)
                                if sl.mode == "RGBA":
                                    alpha = np.array(sl)[..., 3]
                                    opaque = alpha > 0
                                    total = int(opaque.sum())
                                    if total == 0:
                                        return 1.0
                                    return int((white_pixels & opaque).sum()) / total
                                else:
                                    total = arr.shape[0] * arr.shape[1]
                                    if total == 0:
                                        return 1.0
                                    return int(white_pixels.sum()) / total
                                    
                            ratio_i = get_white_ratio(slice_i)
                            ratio_j = get_white_ratio(slice_j)
                            
                            # Component with LESS white ratio must be drawn ON TOP
                            # If ratio_i < ratio_j - 0.02, component j (more white) must be underneath component i (less white).
                            # So j must be drawn BEFORE i, i.e., j -> i.
                            if ratio_i < ratio_j - 0.02:
                                adj[j].append(i)
                                in_degree[i] += 1
                            elif ratio_j < ratio_i - 0.02:
                                adj[i].append(j)
                                in_degree[j] += 1
                    except Exception:
                        pass

    # Kahn's algorithm with priority queue.
    # To resolve tie-breaks (e.g. non-overlapping elements or equal white ratios),
    # we prioritize drawing LARGER components first.
    # Negative area is used as heapq is a min-heap.
    pq = []
    for i in range(N):
        if in_degree[i] == 0:
            heapq.heappush(pq, (-areas[i], i))

    sorted_indices = []
    while pq:
        _, u = heapq.heappop(pq)
        sorted_indices.append(u)
        for v in adj[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                heapq.heappush(pq, (-areas[v], v))

    # Cycle fallback (safety net): if topological sort output doesn't cover all components,
    # fall back to standard area descending sorting.
    if len(sorted_indices) < N:
        sorted_comps = sorted(
            comps,
            key=lambda c: (c.bbox_int[2] - c.bbox_int[0]) * (c.bbox_int[3] - c.bbox_int[1]),
            reverse=True
        )
    else:
        sorted_comps = [comps[idx] for idx in sorted_indices]
    
    for comp in sorted_comps:
        crop_path = os.path.join(sample.components_dir, comp.filename)
        if not os.path.exists(crop_path):
            continue
            
        x0, y0, x1, y1 = comp.bbox_int
        x0 = max(0, x0); y0 = max(0, y0)
        x1 = min(W, x1); y1 = min(H, y1)
        bw = x1 - x0; bh = y1 - y0
        if bw <= 0 or bh <= 0:
            continue
            
        with Image.open(crop_path) as im:
            # Convert to RGBA to preserve the alpha channel/transparency if available
            crop_rgba = im.convert("RGBA")
            if comp.segmentation_mode in ("mask", "polygon"):
                crop_rgba = fill_alpha_holes(crop_rgba)
            
        # Resize crop to fit bbox if dimensions disagree
        if crop_rgba.size != (bw, bh):
            crop_rgba = crop_rgba.resize((bw, bh), Image.LANCZOS)
            
        # Composite crop onto the canvas using alpha blending
        canvas_pil = Image.fromarray(canvas[y0:y1, x0:x1])
        canvas_pil.paste(crop_rgba, (0, 0), crop_rgba)
        canvas[y0:y1, x0:x1] = np.array(canvas_pil.convert("RGB"))
        
    return canvas


def content_mask(original_rgb: np.ndarray, bg_tolerance: int = 12) -> np.ndarray:
    """Estimate the content mask: pixels that meaningfully differ from background.
    Background color is estimated from the median of the 4-px border."""
    arr = original_rgb
    H, W = arr.shape[:2]
    border = np.concatenate(
        [
            arr[:4, :, :].reshape(-1, 3),
            arr[-4:, :, :].reshape(-1, 3),
            arr[:, :4, :].reshape(-1, 3),
            arr[:, -4:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    bg = np.median(border, axis=0).astype(np.int16)
    diff = np.abs(arr.astype(np.int16) - bg[None, None, :]).max(axis=-1)
    return diff > bg_tolerance


def union_bbox_mask(sample: SlideEvalSample) -> np.ndarray:
    """Bool mask, True where any final-component bbox covers a pixel."""
    W, H = sample.image_size
    mask = np.zeros((H, W), dtype=bool)
    for comp in sample.final_components:
        x0, y0, x1, y1 = comp.bbox_int
        x0 = max(0, x0); y0 = max(0, y0)
        x1 = min(W, x1); y1 = min(H, y1)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = True
    return mask


def bbox_coverage_count(sample: SlideEvalSample) -> np.ndarray:
    """Int array, counts how many bboxes cover each pixel."""
    W, H = sample.image_size
    counts = np.zeros((H, W), dtype=np.int32)
    for comp in sample.final_components:
        x0, y0, x1, y1 = comp.bbox_int
        x0 = max(0, x0); y0 = max(0, y0)
        x1 = min(W, x1); y1 = min(H, y1)
        if x1 > x0 and y1 > y0:
            counts[y0:y1, x0:x1] += 1
    return counts
