"""M3: Per-component VLM judge (PPTAgent-style).

For each crop in `final/components/`, ask the VLM:
  cohesion (1-5)     : is this one coherent semantic unit?
  label_fit (1-5)    : does the predicted `text_type` describe what's shown?

The component crops are tightly bounded so the model sees only the predicted unit.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from .common import SlideEvalSample


_SYSTEM = (
    "You are an evaluation judge for a slide-decomposition system. "
    "You receive one cropped image at a time and a label the system assigned to it. "
    "Rate strictly on two axes from 1 to 5 and return JSON only."
)


_USER_TEMPLATE = (
    "The cropped image is the system's prediction of one slide component.\n"
    "The system labelled it: \"{label}\".\n\n"
    "Rate:\n"
    "  cohesion (1-5): Is the crop a single coherent semantic unit? "
    "5 = clean single unit (e.g. one chart, one title, one merged flowchart). "
    "3 = mostly fine but with stray fragments. 1 = multiple unrelated things or noise.\n"
    "  label_fit (1-5): Does the label describe the dominant content? "
    "5 = perfect. 3 = vague but not wrong. 1 = wrong.\n\n"
    "Return JSON: {{\"cohesion\": int, \"label_fit\": int, \"reason\": short string}}."
)


_SCHEMA = {
    "type": "object",
    "properties": {
        "cohesion": {"type": "integer", "minimum": 1, "maximum": 5},
        "label_fit": {"type": "integer", "minimum": 1, "maximum": 5},
        "reason": {"type": "string"},
    },
    "required": ["cohesion", "label_fit", "reason"],
}


def evaluate(sample: SlideEvalSample, vlm_client) -> dict[str, Any]:
    per_component: list[dict[str, Any]] = []
    for comp in sample.final_components:
        crop_path = os.path.join(sample.components_dir, comp.filename)
        if not os.path.exists(crop_path):
            per_component.append({
                "filename": comp.filename,
                "error": "crop missing",
            })
            continue
        user = _USER_TEMPLATE.format(label=comp.text_type or "(no label)")
        try:
            verdict = vlm_client.judge_json(
                system=_SYSTEM,
                user=user,
                images=[crop_path],
                schema=_SCHEMA,
            )
        except Exception as e:
            per_component.append({
                "filename": comp.filename,
                "error": f"{type(e).__name__}: {e}",
            })
            continue
        per_component.append({
            "filename": comp.filename,
            "label": comp.text_type,
            "is_merged": comp.is_merged,
            "cohesion": verdict.get("cohesion"),
            "label_fit": verdict.get("label_fit"),
            "reason": verdict.get("reason"),
        })

    cohesions = [c["cohesion"] for c in per_component if isinstance(c.get("cohesion"), (int, float))]
    label_fits = [c["label_fit"] for c in per_component if isinstance(c.get("label_fit"), (int, float))]
    return {
        "per_component": per_component,
        "mean_cohesion": float(np.mean(cohesions)) if cohesions else None,
        "mean_label_fit": float(np.mean(label_fits)) if label_fits else None,
        "num_judged": len(cohesions),
        "num_failed": sum(1 for c in per_component if c.get("error")),
    }


def aggregate(slide_results: list[dict[str, Any]]) -> dict[str, Any]:
    all_cohesion: list[float] = []
    all_label_fit: list[float] = []
    for r in slide_results:
        for c in r.get("per_component", []):
            if isinstance(c.get("cohesion"), (int, float)):
                all_cohesion.append(c["cohesion"])
            if isinstance(c.get("label_fit"), (int, float)):
                all_label_fit.append(c["label_fit"])
    out: dict[str, Any] = {}
    if all_cohesion:
        out["mean_cohesion"] = float(np.mean(all_cohesion))
        out["std_cohesion"] = float(np.std(all_cohesion))
    if all_label_fit:
        out["mean_label_fit"] = float(np.mean(all_label_fit))
        out["std_label_fit"] = float(np.std(all_label_fit))
    out["n_components_judged"] = len(all_cohesion)
    return out
