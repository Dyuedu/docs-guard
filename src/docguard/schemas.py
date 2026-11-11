"""Response schema shared by the Triton pipeline, the FastAPI gateway and tests."""

from __future__ import annotations

from pydantic import BaseModel, Field, conlist

BBox = conlist(int, min_length=4, max_length=4)  # [x1, y1, x2, y2] in original image pixels


class FieldValue(BaseModel):
    value: str | None = None
    confidence: float = Field(0.0, ge=0.0, le=1.0)


class TamperingResult(BaseModel):
    tampered_score: float = Field(..., ge=0.0, le=1.0)
    tampered_area_ratio: float = Field(..., ge=0.0, le=1.0)
    tampered_bboxes: list[BBox] = Field(default_factory=list)
    overlaps_fields: list[str] = Field(default_factory=list)


class ExtractedFields(BaseModel):
    seller: FieldValue = Field(default_factory=FieldValue)
    timestamp: FieldValue = Field(default_factory=FieldValue)
    total_amount: FieldValue = Field(default_factory=FieldValue)


class LatencyBreakdown(BaseModel):
    preprocess: float = 0.0
    tampering: float = 0.0
    ocr: float = 0.0
    total: float = 0.0


class VerifyResponse(BaseModel):
    request_id: str
    is_authentic: bool
    decision_reason: str
    tampering_result: TamperingResult
    extracted_fields: ExtractedFields
    latency_ms: LatencyBreakdown
