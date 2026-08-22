"""M4: Holistic VLM judge (REFLEX / PPTEval style).

Show the original slide and the canvas-composited reconstruction side-by-side;
ask the VLM to rate three axes from 1 to 5:

  coverage           — does the reconstruction include every meaningful element?
  granularity        — are merges correct? are anything overly fragmented or overly lumped?
  non_redundancy     — does the decomposition avoid double-counting / overlap?

This is the headline semantic-quality number per metrics_survey.md.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .common import SlideEvalSample, FinalComponent, build_reconstruction


_SYSTEM = (
    "You are an evaluation judge for slide-decomposition. You will see two images:\n"
    "1. The ORIGINAL slide annotated with numbered, colored bounding boxes (e.g. #1, #2).\n"
    "2. The RECONSTRUCTION built by pasting each predicted component crop onto a blank canvas at its predicted bbox, annotated with matching numbered, colored bounding boxes.\n"
    "Use these visual annotations to judge the decomposition's coverage, granularity, and redundancy. "
    "CRITICAL RULES FOR OVERLAPS AND FRAGMENTATION:\n"
    "- Be TOLERANT of fragmentation and overlaps if they serve the purpose of \"editability\" in presentation software. For example, separating text from its background container shape, or decomposing complex graphics into layered nested shapes, is highly desirable for editing and should NOT be penalized.\n"
    "- ONLY penalize UNNECESSARY fragmentation (e.g., a single continuous sentence broken into individual words, or a coherent flowchart shattered into meaningless unrecognizable bits) and UNNECESSARY redundancy (e.g., completely duplicate cloned boxes capturing the exact same content multiple times, or bad layouts that visually occlude distinct text).\n"
    "In your JSON response, always refer to specific components by their exact IDs (e.g., '#3 overlaps with #5', or '#1 is a merged chart title') in the 'reason' field. "
    "Return JSON only."
)


_USER = (
    "Image 1 is the original slide with numbered bounding boxes. "
    "Image 2 is the reconstruction canvas with matching numbered bounding boxes.\n"
    "The predicted components list is:\n"
    "{components}\n\n"
    "Rate each axis 1-5:\n"
    "  coverage (1-5): Are all meaningful slide elements present in the reconstruction? "
    "5 = nothing important missing. 1 = many elements absent.\n"
    "  granularity (1-5): Are the chosen units the right grain? 5 = grain is perfect. 1 = unnecessarily overly fragmented or wrongly fused. NOTE: Be forgiving of fragmentation if it separates text from shapes or creates editable nested graphical layers. Only penalize senseless shattering (e.g. single sentences broken into words, or unified charts broken into 50 tiny uneditable lines).\n"
    "  non_redundancy (1-5): Does the system avoid duplicate components and erroneous overlaps? 5 = no erroneous redundancy. 1 = many duplicates or severe layout occlusions. NOTE: Do NOT penalize overlaps that represent logical layering (e.g. text layered over a background box, or nested graphics). Only penalize actual duplicate detections of the identical visual content or destructive occlusions.\n\n"
    "Return JSON: {{\"coverage\": int, \"granularity\": int, \"non_redundancy\": int, "
    "\"reason\": short string (<80 words, must cite specific visual component IDs like #1, #2 to justify your ratings)}}."
)


_SCHEMA = {
    "type": "object",
    "properties": {
        "coverage": {"type": "integer", "minimum": 1, "maximum": 5},
        "granularity": {"type": "integer", "minimum": 1, "maximum": 5},
        "non_redundancy": {"type": "integer", "minimum": 1, "maximum": 5},
        "reason": {"type": "string"},
    },
    "required": ["coverage", "granularity", "non_redundancy", "reason"],
}


def _load_font(size: int = 15) -> Any:
    # Try a few common font paths on Linux
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]
    for p in font_paths:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _draw_bboxes(img: Image.Image, components: list[FinalComponent]) -> Image.Image:
    """Draw numbered, color-coded bounding boxes on a PIL image.
    
    Draws rectangular outlines and visual label tags (e.g. #1, #2) inside the top-left of boxes.
    """
    img_copy = img.copy().convert("RGB")
    draw = ImageDraw.Draw(img_copy)
    
    # Dynamically scale font size and box stroke width based on slide dimensions
    # For a 1280px wide slide, font_size will be around 28px, box_width = 3px
    # For a 1920px wide slide, font_size will be around 43px, box_width = 5px
    font_size = max(24, int(round(img_copy.width / 45)))
    box_width = max(3, int(round(img_copy.width / 400)))
    font = _load_font(font_size)
    
    # Modern professional colors (Tailwind CSS 500 palette)
    colors = [
        (239, 68, 68),   # Red
        (34, 197, 94),   # Green
        (59, 130, 246),  # Blue
        (168, 85, 247),  # Purple
        (234, 179, 8),   # Yellow
        (249, 115, 22),  # Orange
        (6, 182, 212),   # Cyan
        (236, 72, 153),  # Pink
    ]
    
    for i, comp in enumerate(components):
        color = colors[i % len(colors)]
        x0, y0, x1, y1 = comp.bbox_int
        
        # Clamp coordinates to image boundaries to prevent drawing out of bounds
        x0 = max(0, min(x0, img_copy.width - 1))
        y0 = max(0, min(y0, img_copy.height - 1))
        x1 = max(0, min(x1, img_copy.width - 1))
        y1 = max(0, min(y1, img_copy.height - 1))
        
        if x1 <= x0 or y1 <= y0:
            continue
            
        # Draw bounding box outline
        draw.rectangle([x0, y0, x1, y1], outline=color, width=box_width)
        
        # Format visual label tag
        label = f"#{i + 1}"
        
        # Safely measure text size across different Pillow versions
        try:
            # Modern Pillow (>= 8.0.0)
            l, t, r, b = draw.textbbox((0, 0), label, font=font)
            text_w = r - l
            text_h = b - t
        except AttributeError:
            try:
                # Older Pillow fallback
                text_w, text_h = draw.textsize(label, font=font)
            except AttributeError:
                # Absolute fallback scaled with font_size
                text_w = len(label) * int(font_size * 0.6)
                text_h = int(font_size * 1.1)
                
        # Draw a beautiful background tag rectangle
        pad_x = max(4, int(font_size * 0.2))
        pad_y = max(2, int(font_size * 0.1))
        bg_w = text_w + 2 * pad_x
        bg_h = text_h + 2 * pad_y
        
        # Check if the bounding box is too small to contain the tag inside it without substantial occlusion
        box_w = x1 - x0
        box_h = y1 - y0
        is_small = (box_w < bg_w * 1.5) or (box_h < bg_h * 1.5)
        
        if is_small:
            # Place outside: try above the top-left corner first
            if y0 - bg_h >= 0:
                lx0 = max(0, min(x0, img_copy.width - bg_w))
                ly0 = y0 - bg_h
            # If not enough space above, try placing it just below the bottom-left corner
            elif y1 + bg_h <= img_copy.height:
                lx0 = max(0, min(x0, img_copy.width - bg_w))
                ly0 = y1
            # Fallback to inside top-left if both above and below leak out of the image boundaries
            else:
                lx0 = max(0, min(x0, img_copy.width - bg_w))
                ly0 = max(0, min(y0, img_copy.height - bg_h))
        else:
            # Position label inside top-left corner, ensuring it doesn't leak out of the image
            lx0 = max(0, min(x0, img_copy.width - bg_w))
            ly0 = max(0, min(y0, img_copy.height - bg_h))
            
        lx1 = lx0 + bg_w
        ly1 = ly0 + bg_h
        
        draw.rectangle([lx0, ly0, lx1, ly1], fill=color)
        draw.text((lx0 + pad_x, ly0 + pad_y), label, fill=(255, 255, 255), font=font)
        
    return img_copy


def evaluate(
    sample: SlideEvalSample,
    vlm_client,
    save_reconstruction_to: str | None = None,
) -> dict[str, Any]:
    # 1. Draw bounding boxes on original slide copy and save to temp file
    original_img = Image.open(sample.original_image_path)
    annotated_orig = _draw_bboxes(original_img, sample.final_components)
    
    tmp_orig = tempfile.NamedTemporaryFile(suffix="_orig_annotated.png", delete=False)
    orig_annotated_path = tmp_orig.name
    tmp_orig.close()
    annotated_orig.save(orig_annotated_path)
    
    # 2. Build reconstruction canvas and draw bounding boxes
    recon = build_reconstruction(sample)
    recon_img = Image.fromarray(recon)
    annotated_recon = _draw_bboxes(recon_img, sample.final_components)
    
    if save_reconstruction_to:
        os.makedirs(os.path.dirname(save_reconstruction_to), exist_ok=True)
        recon_path = save_reconstruction_to.replace("_recon.png", "_recon_annotated.png")
        if recon_path == save_reconstruction_to:
            recon_path += "_annotated.png"
        annotated_recon.save(recon_path)
        
        orig_annotated_path_save = save_reconstruction_to.replace("_recon.png", "_orig_annotated.png")
        if orig_annotated_path_save == save_reconstruction_to:
            orig_annotated_path_save += "_orig_annotated.png"
        annotated_orig.save(orig_annotated_path_save)
        
        keep = True
    else:
        tmp_recon = tempfile.NamedTemporaryFile(suffix="_recon_annotated.png", delete=False)
        recon_path = tmp_recon.name
        tmp_recon.close()
        annotated_recon.save(recon_path)
        keep = False

    comp_listing = "\n".join(
        f"#{i+1}: Type={c.text_type or '?'}, IsMerged={c.is_merged}, Bbox={c.bbox_int}"
        for i, c in enumerate(sample.final_components)
    )
    user = _USER.format(components=comp_listing or "(none)")
    
    try:
        verdict = vlm_client.judge_json(
            system=_SYSTEM,
            user=user,
            images=[orig_annotated_path, recon_path],
            schema=_SCHEMA,
        )
    except Exception as e:
        verdict = {"error": f"{type(e).__name__}: {e}"}
    finally:
        # Clean up temporary original image
        if os.path.exists(orig_annotated_path):
            try:
                os.unlink(orig_annotated_path)
            except OSError:
                pass
        # Clean up temporary reconstruction image if not keeping
        if not keep and os.path.exists(recon_path):
            try:
                os.unlink(recon_path)
            except OSError:
                pass

    out = {
        "coverage": verdict.get("coverage"),
        "granularity": verdict.get("granularity"),
        "non_redundancy": verdict.get("non_redundancy"),
        "reason": verdict.get("reason"),
        "num_final_components": len(sample.final_components),
    }
    if "error" in verdict:
        out["error"] = verdict["error"]
    if save_reconstruction_to:
        out["reconstruction_path"] = save_reconstruction_to
    return out


def aggregate(slide_results: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ["coverage", "granularity", "non_redundancy"]
    out: dict[str, Any] = {}
    for k in keys:
        vals = [r[k] for r in slide_results if isinstance(r.get(k), (int, float))]
        if vals:
            out[f"mean_{k}"] = float(np.mean(vals))
            out[f"std_{k}"] = float(np.std(vals))
    out["n_slides_judged"] = sum(
        1 for r in slide_results if isinstance(r.get("coverage"), (int, float))
    )
    out["n_failed"] = sum(1 for r in slide_results if r.get("error"))
    return out
