"""Intent 002 — corpus + synthetic text lines (spec AC-1, AC-2, AC-5, AC-6).

Expected values come from intent/002-synth-ocr-data/spec.md.
"""

import random
from pathlib import Path

import numpy as np
import pytest

from docguard.data.synth_text import LineRenderer, ReceiptCorpus, filter_charset, load_vocab_set

TEST_FONT = str(Path(__file__).parent / "fonts" / "BeVietnamPro-Regular.ttf")
VOCAB = Path(__file__).resolve().parents[1] / "configs" / "vocab_vi.txt"

TRAIN_TEXTS = [
    "MINIMART ANAN", "Chợ Sủi Phú Thị Gia Lâm", "Ngày : 11/08/2020 08:06", "Tổng tiền:", "74,000",
    "VM+ QNH 590 Trần Phú", "P Cẩm Bình TP Cẩm Phả QNH", "TỔNG TIỀN PHẢI T.TOÁN", "Ngày bán: 15/08/2020 11:41",
    "Số 112 Thanh Niên, P. Cẩm Thành", "ĐỊA CHỈ ĐẠI LÝ:", "CO.OP MART", "125.000", "Siêu thị Vinmart Hà Đông",
]
VAL_TEST_TEXTS = ["PHỞ MŨ", "Tổ 7 khu Tân Lập 4 - P Cẩm Thủy", "57,000.00", "Ngày: 23/09/2019 11:00:11"]


@pytest.fixture(scope="module")
def vocab():
    return load_vocab_set(VOCAB)


@pytest.fixture(scope="module")
def corpus(vocab):
    return ReceiptCorpus(TRAIN_TEXTS, vocab, seed=0)


# --- AC-1: charset ------------------------------------------------------------------------------

def test_filter_charset_drops_out_of_vocab_chars(vocab):
    assert filter_charset("Tổng: 125.000₫ ©", vocab) == "Tổng: 125.000 "
    assert filter_charset("abc", {"a", "b"}) == "ab"


def test_corpus_charset_subset_of_vocab(corpus, vocab):
    allowed = vocab | {" "}
    rng = random.Random(1)
    for _ in range(500):
        line = corpus.line(rng)
        assert line, "empty line"
        assert set(line) <= allowed, set(line) - allowed
    for _ in range(50):
        for text, label in corpus.receipt(rng):
            assert set(text) <= allowed, set(text) - allowed


def test_receipt_structure_and_labels(corpus):
    rng = random.Random(2)
    labels = {"SELLER", "ADDRESS", "TIMESTAMP", "TOTAL_COST", "OTHER"}
    for _ in range(20):
        rows = corpus.receipt(rng)
        assert len(rows) >= 8                                   # spec AC-3 minimum lines
        seen = {label for _, label in rows}
        assert seen <= labels
        assert {"SELLER", "TIMESTAMP", "TOTAL_COST"} <= seen     # fields the KIE rules extract
        assert all(text.strip() for text, _ in rows)


# --- AC-2: no leak from val/test texts ----------------------------------------------------------

def test_corpus_no_val_test_leak(vocab):
    c = ReceiptCorpus(TRAIN_TEXTS, vocab, seed=0)
    rng = random.Random(3)
    lines = {c.line(rng) for _ in range(2000)}
    forbidden = {ReceiptCorpus.normalize(t) for t in VAL_TEST_TEXTS if len(t) > 3}
    assert not {ReceiptCorpus.normalize(l) for l in lines} & forbidden


def test_normalize_collapses_space_and_case():
    assert ReceiptCorpus.normalize("  Tổng   TIỀN : ") == "tổng tiền :"


# --- AC-5: line image contract -------------------------------------------------------------------

def test_line_image_shape_and_label(corpus):
    r = LineRenderer([TEST_FONT], height=32, max_width=320)
    rng = random.Random(4)
    non_blank = 0
    for _ in range(100):
        text = corpus.line(rng)[:40]
        img = r.render(text, rng)
        assert img.dtype == np.uint8 and img.ndim == 3 and img.shape[2] == 3
        assert img.shape[0] == 32
        assert 1 <= img.shape[1] <= 320
        assert len(text) <= 40
        non_blank += bool(text.strip())
    assert non_blank >= 95


def test_line_renderer_long_text_is_shrunk_not_cropped():
    r = LineRenderer([TEST_FONT], height=32, max_width=320)
    img = r.render("Tổng tiền phải thanh toán: 1.250.000 VND", random.Random(0))
    assert img.shape[1] == 320  # fills the full width after proportional shrink


def test_line_renderer_has_ink():
    r = LineRenderer([TEST_FONT], height=32, max_width=320)
    img = r.render("125.000", random.Random(0))
    assert img.std() > 10  # not a flat image


# --- AC-6: determinism ----------------------------------------------------------------------------

def test_corpus_deterministic(vocab):
    a = ReceiptCorpus(TRAIN_TEXTS, vocab, seed=7)
    b = ReceiptCorpus(TRAIN_TEXTS, vocab, seed=7)
    ra, rb = random.Random(5), random.Random(5)
    assert [a.line(ra) for _ in range(50)] == [b.line(rb) for _ in range(50)]
    assert a.receipt(ra) == b.receipt(rb)


def test_line_render_deterministic():
    r1 = LineRenderer([TEST_FONT]); r2 = LineRenderer([TEST_FONT])
    a = r1.render("Ngày: 12/08/2020", random.Random(9))
    b = r2.render("Ngày: 12/08/2020", random.Random(9))
    assert np.array_equal(a, b)


# --- AC-2 (amended): provenance-based leak check ---------------------------------------------------

from scripts.synth_ocr_report import check_provenance  # noqa: E402


def test_provenance_check_accepts_train_only_and_shared_lines():
    train = ["MINIMART ANAN", "Tổng tiền:", "10,000"]
    held = ["10,000", "PHỞ MŨ"]                       # "10,000" also in train -> legitimate
    corpus = ["MINIMART ANAN", "tổng   TIỀN:", "10,000"]  # normalisation: case + spaces
    leaked, info = check_provenance(corpus, train, held)
    assert leaked == []
    assert info["heldout_overlap_ratio"] == pytest.approx(0.5)   # 1 of 2 held-out texts also in train


def test_provenance_check_flags_line_not_from_train():
    train = ["MINIMART ANAN"]
    held = ["PHỞ MŨ"]
    leaked, _ = check_provenance(["MINIMART ANAN", "PHỞ MŨ"], train, held)
    assert leaked == ["PHỞ MŨ"]


def test_provenance_check_flags_any_non_train_line():
    # a corpus line found neither in train nor in held-out is still not train-sourced
    leaked, _ = check_provenance(["CỬA HÀNG LẠ"], ["MINIMART ANAN"], ["PHỞ MŨ"])
    assert leaked == ["CỬA HÀNG LẠ"]
