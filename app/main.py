from __future__ import annotations
import os
import tempfile
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse

from .video import load_video_frames
from .model import VLMService

app = FastAPI(title="Temporal VLM API")
vlm: VLMService | None = None

@app.on_event("startup")
def _startup():
    global vlm
    use_cuda = os.environ.get("USE_CUDA", "1") == "1"
    vlm = VLMService(device="cuda" if use_cuda else "cpu")

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    clip_id: str = Form("unknown_clip"),
    num_frames: int = Form(8),
):
    assert vlm is not None

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        frames = load_video_frames(tmp_path, num_frames=num_frames)
        pred = vlm.predict(frames, clip_id=clip_id)
        return JSONResponse(pred)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass