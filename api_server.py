from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator

from video_processor import process_video_to_segments


app = FastAPI(title="Video OCR Processor API", version="1.0.0")


class VideoProcessRequest(BaseModel):
    video_path: str
    scan_step: int = 4
    crop: Optional[List[int]] = None  # [x1, y1, x2, y2]

    @validator("scan_step")
    def validate_scan_step(cls, value: int) -> int:
        if value < 1:
            raise ValueError("scan_step must be >= 1")
        return value

    @validator("crop")
    def validate_crop(cls, value: Optional[List[int]]) -> Optional[List[int]]:
        if value is not None:
            if len(value) != 4:
                raise ValueError("crop must contain exactly 4 integers [x1, y1, x2, y2]")
        return value


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/process")
def process_video(request: VideoProcessRequest):
    video_path = Path(request.video_path)
    if not video_path.exists():
        raise HTTPException(status_code=400, detail=f"Video not found: {request.video_path}")

    crop = tuple(request.crop) if request.crop else None

    try:
        segments = process_video_to_segments(
            video_path=str(video_path),
            scan_step=request.scan_step,
            crop=crop,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "video_path": str(video_path),
        "segment_count": len(segments),
        "segments": segments,
    }

