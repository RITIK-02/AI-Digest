import pytest
from pydantic import ValidationError

from src.models import LLMRequest
from src.pipeline.triage import decode_custom_id, encode_custom_id


def test_triage_custom_id_round_trips():
    ids = [10436, 11681, 11934, 0, 35, 36, 999_999]
    assert decode_custom_id(encode_custom_id(ids)) == ids


def test_triage_custom_id_fits_vendor_limit_for_full_batch():
    # Ten 6-digit ids — the case decimal encoding overflows.
    custom_id = encode_custom_id([999_999 - i for i in range(10)])
    LLMRequest(custom_id=custom_id, prompt="x")  # raises if invalid
    assert len(custom_id) <= 64


def test_triage_decodes_legacy_format():
    assert decode_custom_id("triage:10436,11681") == [10436, 11681]


@pytest.mark.parametrize("bad", ["brief:today", "triage:1,2", "a" * 65, ""])
def test_llm_request_rejects_ids_vendors_reject(bad):
    with pytest.raises(ValidationError):
        LLMRequest(custom_id=bad, prompt="x")
