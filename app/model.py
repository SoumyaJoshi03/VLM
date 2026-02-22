from __future__ import annotations
import json
import re
from typing import Any, Dict, List

import torch
from PIL import Image
from transformers import Qwen2VLProcessor, Qwen2VLForConditionalGeneration
from transformers import AutoModelForCausalLM
from transformers import Qwen2VLProcessor

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

OPS = [
    "Box Setup", "Inner Packing", "Tape", "Put Items", "Pack",
    "Wrap", "Label", "Final Check", "Idle", "Unknown",
]

def _extract_json(text: str) -> Dict[str, Any]:
    m = _JSON_RE.search(text)
    if not m:
        raise ValueError(f"Model did not return JSON. Raw:\n{text[:500]}")
    return json.loads(m.group(0))

def _to_pil_list(frames_rgb: List) -> List[Image.Image]:
    pil = []
    for fr in frames_rgb:
        # fr is typically np.ndarray HxWx3 uint8
        if isinstance(fr, Image.Image):
            pil.append(fr.convert("RGB"))
        else:
            pil.append(Image.fromarray(fr).convert("RGB"))
    return pil

class VLMService:
    def __init__(self, device: str = "cuda"):
        self.device = device

        # Avoid AutoProcessor video-processing auto detection issues
        self.processor = Qwen2VLProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16 if "cuda" in device else torch.float32,
            device_map="auto" if "cuda" in device else None,  # cpu: keep None
            trust_remote_code=True,
        )
        self.model.eval()

    @torch.inference_mode()
    def predict(self, frames_rgb: List, clip_id: str) -> Dict[str, Any]:
        images = _to_pil_list(frames_rgb)

        system = (
            "You are a warehouse video analyst. Return ONLY valid JSON with keys: "
            "clip_id, dominant_operation, temporal_segment{start_frame,end_frame}, "
            "anticipated_next_operation, confidence."
        )

        user = (
            f"Analyze this clip and output JSON only.\n"
            f"clip_id={clip_id}\n"
            f"Operations: {', '.join(OPS)}\n"
            f"Use frame indices within the clip timeline.\n"
        )

        # Qwen2-VL expects a chat format; include images as part of the content.
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": [{"type": "image", "image": img} for img in images] + [{"type": "text", "text": user}]},
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.processor(
            text=[text],
            images=images,
            return_tensors="pt",
        ).to(self.model.device)

        out = self.model.generate(**inputs, max_new_tokens=256)
        decoded = self.processor.batch_decode(out, skip_special_tokens=True)[0]

        data = _extract_json(decoded)

        # Minimal schema enforcement
        data["clip_id"] = data.get("clip_id", clip_id)
        data.setdefault("dominant_operation", "Unknown")
        data.setdefault("temporal_segment", {"start_frame": 0, "end_frame": max(0, len(frames_rgb) - 1)})
        data.setdefault("anticipated_next_operation", "Unknown")
        data["confidence"] = float(data.get("confidence", 0.0))

        if data["dominant_operation"] not in OPS:
            data["dominant_operation"] = "Unknown"
        if data["anticipated_next_operation"] not in OPS:
            data["anticipated_next_operation"] = "Unknown"

        return data