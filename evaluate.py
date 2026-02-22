from __future__ import annotations
import argparse
import json
from typing import Dict, Tuple, Any

def iou_1d(a0: int, a1: int, b0: int, b1: int) -> float:
    inter = max(0, min(a1, b1) - max(a0, b0) + 1)
    union = (a1 - a0 + 1) + (b1 - b0 + 1) - inter
    return inter / union if union > 0 else 0.0

def eval_one(pred: Dict[str, Any], gt: Dict[str, Any]) -> Tuple[bool, bool, bool]:
    oca = pred.get("dominant_operation") == gt.get("dominant_operation")
    aa = pred.get("anticipated_next_operation") == gt.get("anticipated_next_operation")

    ps = pred.get("temporal_segment") or {}
    gs = gt.get("temporal_segment") or {}
    ok = all(k in ps for k in ["start_frame", "end_frame"]) and all(k in gs for k in ["start_frame", "end_frame"])
    tiou = False
    if ok:
        tiou = iou_1d(ps["start_frame"], ps["end_frame"], gs["start_frame"], gs["end_frame"]) >= 0.5
    return oca, tiou, aa

def run_eval(pred_path: str, gt_path: str) -> Dict[str, float]:
    preds = json.load(open(pred_path, "r"))
    gts = json.load(open(gt_path, "r"))
    gt_map = {x["clip_id"]: x for x in gts}

    oca_c = tiou_c = aa_c = n = 0
    for p in preds:
        cid = p.get("clip_id")
        if cid not in gt_map:
            continue
        o, t, a = eval_one(p, gt_map[cid])
        oca_c += int(o)
        tiou_c += int(t)
        aa_c += int(a)
        n += 1

    return {
        "OCA": oca_c / max(1, n),
        "tIoU@0.5": tiou_c / max(1, n),
        "AA@1": aa_c / max(1, n),
        "N": n,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()

    scores = run_eval(args.pred, args.gt)
    out = {"scores": scores}

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()