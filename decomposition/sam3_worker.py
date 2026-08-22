"""Persistent SAM3 worker for Agent E (experiment h8-persistent-sam3-worker).

Loads the SAM3 model ONCE, then serves segmentation jobs from a file-based queue,
eliminating the ~140 s per-invocation fixed cost (torch import + 6.7 GB checkpoint
load) that dominates pipeline wall-clock.

Queue protocol (dir = $SAM3_WORKER_DIR):
  client writes  job_<id>.json    {"id": ..., "argv": [<E-script CLI args...>]}
  worker writes  done_<id>.json   {"id": ..., "returncode": 0}            on success
                                  {"id": ..., "returncode": 1, "error": tb} on failure
  client deletes both when consumed. "argv" is EXACTLY what would follow the script
  path in the legacy subprocess call, so arg parsing is byte-for-byte the script's own
  parse_args(). A job whose --ckpt/--base_ckpt/--device differ from the worker's is
  rejected (returncode 2) — one worker serves one model on one device.

Run (in the environment that has torch + the sam3 package installed, pinned to
the GPU the pipeline uses):
  python decomposition/sam3_worker.py \
      --script sam3/infer_remove_components_overlap_priority.py \
      --ckpt <decoder.pt> --base_ckpt <sam3.pt> --device cuda:0 \
      --queue_dir /tmp/sam3_worker_queue

Requires `run(args, model=None)` in the E script (falls back to loading itself when
model is None, so the legacy subprocess path is unchanged).

Stop: write a file named `stop` into the queue dir, or SIGTERM.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path


def load_module(script_path: Path):
    spec = importlib.util.spec_from_file_location("sam3_e_script", str(script_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sam3_e_script"] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", required=True, help="Path to infer_remove_components_overlap_priority.py")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base_ckpt", required=True)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--queue_dir", required=True)
    ap.add_argument("--poll_sec", type=float, default=0.2)
    args = ap.parse_args()

    queue = Path(args.queue_dir)
    queue.mkdir(parents=True, exist_ok=True)

    mod = load_module(Path(args.script).resolve())

    import torch  # after module import (module imports it anyway)
    device = torch.device(args.device)
    print(f"[worker] Loading model once on {device} ...", flush=True)
    t0 = time.time()
    model = mod.build_sam3_image_model(
        device=str(device),
        eval_mode=True,
        checkpoint_path=args.base_ckpt,
        enable_segmentation=True,
    ).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print(f"[worker] fine-tuned ckpt loaded (missing={len(missing)}, unexpected={len(unexpected)}, "
          f"epoch={ckpt.get('epoch')})", flush=True)
    model.eval()
    del ckpt
    print(f"[worker] Ready in {time.time()-t0:.1f}s. Watching {queue}", flush=True)

    def _job_field(argv: list[str], flag: str) -> str | None:
        try:
            return argv[argv.index(flag) + 1]
        except (ValueError, IndexError):
            return None

    while True:
        if (queue / "stop").exists():
            print("[worker] stop file found — exiting.", flush=True)
            return
        jobs = sorted(queue.glob("job_*.json"))
        if not jobs:
            time.sleep(args.poll_sec)
            continue
        for job_path in jobs:
            try:
                job = json.loads(job_path.read_text())
            except (json.JSONDecodeError, OSError):
                time.sleep(0.1)  # writer may still be mid-write
                continue
            jid = job.get("id") or job_path.stem.replace("job_", "")
            done_path = queue / f"done_{jid}.json"
            if done_path.exists():
                continue
            argv = job["argv"]
            result: dict = {"id": jid}
            jc, jb = _job_field(argv, "--ckpt"), _job_field(argv, "--base_ckpt")
            jd = _job_field(argv, "--device")
            if jc != args.ckpt or jb != args.base_ckpt or (jd and jd != args.device):
                result.update(returncode=2,
                              error=f"job ckpt/device mismatch: ckpt={jc} base={jb} dev={jd}")
            else:
                t1 = time.time()
                print(f"[worker] job {jid}: {' '.join(argv[:6])} ...", flush=True)
                old_argv = sys.argv
                try:
                    sys.argv = [str(Path(args.script).resolve())] + argv
                    job_args = mod.parse_args()
                    mod.run(job_args, model=model)
                    result["returncode"] = 0
                except SystemExit as e:  # argparse failure
                    result.update(returncode=1, error=f"SystemExit: {e}")
                except Exception:
                    result.update(returncode=1, error=traceback.format_exc())
                finally:
                    sys.argv = old_argv
                result["wall_sec"] = round(time.time() - t1, 2)
                print(f"[worker] job {jid} done rc={result['returncode']} "
                      f"in {result['wall_sec']}s", flush=True)
            tmp = queue / f".done_{jid}.tmp"
            tmp.write_text(json.dumps(result))
            tmp.rename(done_path)
            try:
                job_path.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    main()
