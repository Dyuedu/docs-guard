import pytest
from pydantic import ValidationError

from docguard.schemas import VerifyResponse


def test_sample_response_is_valid(sample_response):
    r = VerifyResponse.model_validate(sample_response)
    assert r.tampering_result.overlaps_fields == ["total_amount"]
    assert r.extracted_fields.total_amount.value == "125000"


def test_missing_fields_default_to_empty(sample_response):
    del sample_response["extracted_fields"]["seller"]
    r = VerifyResponse.model_validate(sample_response)
    assert r.extracted_fields.seller.value is None


@pytest.mark.parametrize(
    "path,value",
    [
        (("tampering_result", "tampered_score"), 1.5),
        (("tampering_result", "tampered_bboxes"), [[1, 2, 3]]),
        (("extracted_fields", "timestamp", "confidence"), -0.1),
    ],
)
def test_invalid_values_rejected(sample_response, path, value):
    d = sample_response
    for k in path[:-1]:
        d = d[k]
    d[path[-1]] = value
    with pytest.raises(ValidationError):
        VerifyResponse.model_validate(sample_response)
