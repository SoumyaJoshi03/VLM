from __future__ import annotations
import numpy as np
from decord import VideoReader, cpu

def load_video_frames(video_path: str, num_frames: int = 8) -> list[np.ndarray]:
    vr = VideoReader(video_path, ctx=cpu(0))
    total = len(vr)
    if total <= 0:
        raise ValueError("Empty video")

    idx = np.linspace(0, total - 1, num_frames).astype(int)
    frames = vr.get_batch(idx).asnumpy()  
    return [frames[i] for i in range(frames.shape[0])]