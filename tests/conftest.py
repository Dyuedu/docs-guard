import copy

import pytest

SAMPLE_RESPONSE = {
    "request_id": "req_98234",
    "is_authentic": False,
    "decision_reason": "tampered_area_ratio > 0.002",
    "tampering_result": {
        "tampered_score": 0.91,
        "tampered_area_ratio": 0.0041,
        "tampered_bboxes": [[120, 340, 280, 390]],
        "overlaps_fields": ["total_amount"],
    },
    "extracted_fields": {
        "seller": {"value": "CO.OP MART", "confidence": 0.97},
        "timestamp": {"value": "15/08/2021", "confidence": 0.95},
        "total_amount": {"value": "125000", "confidence": 0.88},
    },
    "latency_ms": {"preprocess": 6.1, "tampering": 14.2, "ocr": 18.7, "total": 41.3},
}


@pytest.fixture
def sample_response():
    return copy.deepcopy(SAMPLE_RESPONSE)
