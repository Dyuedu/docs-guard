import cv2
import numpy as np

from docguard.pipeline import processing as P


def test_tile_stitch_roundtrip_covers_every_pixel():
    img = np.random.randint(0, 255, (700, 1100, 3), np.uint8)
    patches, coords = P.tile(img, 512, 384)
    assert patches.shape[1:] == (3, 512, 512)
    ones = np.ones((len(coords), 512, 512), np.float32)
    assert np.allclose(P.stitch(ones, coords, img.shape[:2], 512), 1.0)


def test_tile_small_image_is_padded():
    img = np.zeros((100, 200, 3), np.uint8)
    patches, coords = P.tile(img, 512, 384)
    assert coords == [(0, 0)] and patches.shape == (1, 3, 512, 512)
    assert P.stitch(np.ones((1, 512, 512)), coords, (100, 200), 512).shape == (100, 200)


def test_mask_to_boxes_filters_small_regions():
    prob = np.zeros((100, 100), np.float32)
    prob[10:30, 20:60] = 0.9
    prob[80, 80] = 0.9  # 1 px noise
    assert P.mask_to_boxes(prob, 0.5, min_area=16) == [[20, 10, 60, 30]]


def test_det_postprocess_maps_back_to_image_coords():
    image = np.zeros((480, 960, 3), np.uint8)
    inp, ratio = P.det_preprocess(image, 960)
    assert inp.shape == (1, 3, 960, 960) and ratio == 1.0
    prob = np.zeros((960, 960), np.float32)
    prob[100:120, 200:400] = 0.95
    boxes = P.det_postprocess(prob, ratio, 0.3, 0.6, 1.5, (480, 960))
    assert len(boxes) == 1
    x1, y1, x2, y2 = boxes[0]
    assert x1 < 200 and y1 < 100 and x2 > 400 and y2 > 120  # unclipped outward


def test_crop_for_rec_shape_and_padding():
    img = np.full((100, 300, 3), 255, np.uint8)
    out = P.crop_for_rec(img, [[0, 0, 64, 32], [0, 0, 300, 20]], 32, 320)
    assert out.shape == (2, 3, 32, 320)
    assert np.all(out[0, :, :, 64:] == 0)  # right-padded after resized width 64


def test_ctc_greedy_decode_collapses_repeats_and_blanks():
    vocab = list("ab")
    seq = [1, 1, 0, 1, 2, 2, 0]  # a a _ a b b _  -> "aab"
    logits = np.full((1, len(seq), 3), -10.0, np.float32)
    for t, k in enumerate(seq):
        logits[0, t, k] = 10.0
    (text, conf), = P.ctc_greedy_decode(logits, vocab)
    assert text == "aab" and conf > 0.99


def test_load_vocab_space_token(tmp_path):
    p = tmp_path / "v.txt"
    p.write_text("<space>\na\nb\n", encoding="utf-8")
    assert P.load_vocab(str(p)) == [" ", "a", "b"]


def test_decode_image_downscales_large_input():
    img = np.zeros((3000, 1500, 3), np.uint8)
    ok, buf = cv2.imencode(".png", img)
    out, scale = P.decode_image(buf.tobytes(), 2048)
    assert max(out.shape[:2]) == 2048 and abs(scale - 2048 / 3000) < 1e-6


def test_decide():
    cfg = {"tampered_area_ratio": 0.002, "tampered_score": 0.8}
    assert P.decide(0.1, 0.0, cfg)[0] is True
    assert P.decide(0.1, 0.01, cfg)[0] is False
    assert P.decide(0.95, 0.0, cfg)[0] is False
