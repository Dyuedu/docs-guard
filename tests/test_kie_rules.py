from docguard.kie.rules import OcrLine, extract_fields, normalize_amount, normalize_date


def test_normalize_amount():
    assert normalize_amount("125.000đ") == "125000"
    assert normalize_amount("TỔNG CỘNG: 1,250,000 VND") == "1250000"
    assert normalize_amount("12 500") == "12500"
    assert normalize_amount("abc") is None


def test_normalize_date():
    assert normalize_date("Ngày: 5/8/21 10:32") == "05/08/2021"
    assert normalize_date("15-08-2021") == "15/08/2021"
    assert normalize_date("32/13/2021") is None


def _receipt():
    return [
        OcrLine("CO.OP MART", (100, 20, 500, 80), 0.97),
        OcrLine("123 Lê Lợi, Q1", (100, 90, 450, 115), 0.9),
        OcrLine("Ngày bán: 15/08/2021", (60, 200, 400, 225), 0.95),
        OcrLine("Tổng tiền hàng", (60, 700, 300, 725), 0.93),
        OcrLine("150.000", (500, 701, 620, 724), 0.9),
        OcrLine("Giảm giá 25.000", (60, 740, 620, 765), 0.9),
        OcrLine("Tổng thanh toán", (60, 780, 320, 805), 0.92),
        OcrLine("125.000", (500, 781, 620, 804), 0.88),
        OcrLine("Tiền khách đưa 200.000", (60, 820, 620, 845), 0.9),
    ]


def test_extract_fields_picks_final_total_and_skips_excluded_lines():
    f = extract_fields(_receipt(), image_height=1000)
    assert f["seller"].value == "CO.OP MART"
    assert f["timestamp"].value == "15/08/2021"
    assert f["total_amount"].value == "125000"
    assert f["total_amount"].bbox == (500, 781, 620, 804)


def test_extract_fields_empty():
    f = extract_fields([], image_height=1000)
    assert all(v.value is None for v in f.values())
