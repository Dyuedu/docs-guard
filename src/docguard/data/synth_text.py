"""Receipt-style Vietnamese corpus + synthetic text-line renderer (intent 002).

Design follows synthtiger's text-recognition template (NAVER Corp., MIT,
https://github.com/clovaai/synthtiger/tree/master/examples/synthtiger): corpus -> font ->
colour -> small geometric transform -> post-processing (noise/blur). Re-implemented with
Pillow + OpenCV only, because synthtiger's imgaug dependency does not work with numpy 2.

All randomness comes from the `random.Random` passed in, so output is reproducible.
"""

from __future__ import annotations

import random
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

LABELS = ("SELLER", "ADDRESS", "TIMESTAMP", "TOTAL_COST", "OTHER")
_DIGIT = re.compile(r"\d")
_ADDR_HINT = re.compile(r"(?i)\b(số|đường|phường|p\.|q\.|quận|tp|thành phố|khu|tổ|ngõ|xã|huyện|đc|địa chỉ)\b")

# Hand-written fallback pools (ASCII-safe + Vietnamese), used in addition to MC-OCR train texts.
_ITEMS = [
    "Coca Cola lon 330ml", "Nước suối Lavie 500ml", "Mì Hảo Hảo tôm chua cay", "Sữa tươi Vinamilk 1L",
    "Bánh mì sandwich", "Trứng gà ta 10 quả", "Thịt heo ba chỉ", "Cá basa phi lê", "Rau muống", "Cà chua",
    "Dầu ăn Neptune 1L", "Nước mắm Nam Ngư", "Gạo ST25 5kg", "Đường Biên Hòa 1kg", "Muối i-ốt", "Bột ngọt Ajinomoto",
    "Kem đánh răng PS", "Dầu gội Clear", "Xà phòng Lifebuoy", "Nước rửa chén Sunlight", "Khăn giấy Pulppy",
    "Bia Tiger lon", "Trà xanh 0 độ", "Cà phê sữa đá", "Phở bò tái", "Bún chả", "Cơm gà xối mỡ", "Bánh bao",
    "Sting dâu", "Redbull", "Snack Oishi", "Kẹo Alpenliebe", "Bánh Oreo", "Chocopie", "Sữa chua Vinamilk",
    "Bút bi Thiên Long", "Vở Campus 200 trang", "Pin Panasonic AA", "Khẩu trang y tế", "Nước giặt Omo 3.6kg",
]
_TITLES = ["HÓA ĐƠN BÁN HÀNG", "PHIẾU TÍNH TIỀN", "HOA DON BAN HANG", "HÓA ĐƠN THANH TOÁN", "PHIẾU THANH TOÁN",
           "BILL THANH TOÁN", "HÓA ĐƠN BÁN LẺ"]
_THANKS = ["Cảm ơn quý khách và hẹn gặp lại", "Cam on Quy khach. Hen gap lai!", "Xin cảm ơn quý khách",
           "Chỉ xuất hóa đơn trong ngày", "Tax invoice will be issued within same day", "Quý khách vui lòng giữ hóa đơn",
           "Trân trọng cảm ơn Quý khách", "Hotline: 1900 6868", "Website: www.sieuthi.vn", "Powered by POS365 VN"]
_SELLER_WORDS = ["Siêu thị", "Minimart", "Cửa hàng", "Tạp hóa", "Nhà sách", "Quán", "Shop", "Bách hóa", "Nhà thuốc",
                 "Cafe", "Bakery", "Mart", "Store", "Food", "Market"]
_STREETS = ["Trần Phú", "Nguyễn Trãi", "Lê Lợi", "Hai Bà Trưng", "Thanh Niên", "Cầu Giấy", "Láng Hạ", "Quang Trung",
            "Lý Thường Kiệt", "Điện Biên Phủ", "Võ Thị Sáu", "Trường Chinh", "Nguyễn Văn Cừ", "Phạm Văn Đồng"]
_CITIES = ["Hà Nội", "TP. Hồ Chí Minh", "Đà Nẵng", "Hải Phòng", "Cẩm Phả", "Quảng Ninh", "Cần Thơ", "Nha Trang",
           "Biên Hòa", "Vũng Tàu", "Huế", "Bắc Ninh"]


def load_vocab_set(path: str | Path) -> set[str]:
    with open(path, encoding="utf-8") as f:
        return {line.rstrip("\n") for line in f if line.rstrip("\n")}


def filter_charset(text: str, vocab: set[str]) -> str:
    """Drop every character not in vocab (space is always kept)."""
    return "".join(c for c in text if c == " " or c in vocab)


def heldout_only(train_texts: list[str], heldout_texts: list[str], min_len: int = 4) -> set[str]:
    """Normalised val/test texts (>= min_len chars) that never occur in train (spec AC-2, clause 2)."""
    norm = ReceiptCorpus.normalize
    train = {norm(t) for t in train_texts}
    return {h for h in (norm(t) for t in heldout_texts) if len(h) >= min_len and h not in train}


_MAX_REROLL = 200


class ReceiptCorpus:
    """Generates receipt-like lines/receipts from MC-OCR *train* texts + templates.

    Only texts of the train split may be passed in (spec AC-2). ``blocklist`` holds normalised
    held-out-only texts (see ``heldout_only``); generated lines/receipts that would reproduce one
    are re-drawn, so template output never recreates val/test-only text (AC-2, clause 2).
    """

    def __init__(self, train_texts: list[str], vocab: set[str], seed: int = 0, blocklist: set[str] | None = None):
        self.vocab = vocab
        self.blocklist = frozenset(blocklist or ())
        texts = [filter_charset(t, vocab).strip() for t in train_texts]
        texts = [re.sub(r"\s+", " ", t) for t in texts if t]
        self.sellers = [t for t in texts if not _DIGIT.search(t) and 3 <= len(t) <= 32 and t.upper() == t] or ["CO.OP MART"]
        self.addresses = [t for t in texts if _ADDR_HINT.search(t) and len(t) <= 45] or ["Số 1 Trần Phú"]
        words = {w for t in texts for w in t.split() if len(w) > 1 and not _DIGIT.search(w)}
        self.words = sorted(words) or ["hàng", "tiền"]
        self.items = [filter_charset(x, vocab) for x in _ITEMS]

    # --- primitives ---------------------------------------------------------------------------

    @staticmethod
    def normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip().lower()

    def _ok(self, text: str) -> str:
        return filter_charset(text, self.vocab)

    def blocked(self, text: str) -> bool:
        return bool(self.blocklist) and self.normalize(text) in self.blocklist

    def amount(self, rng: random.Random, big: bool = False) -> str:
        v = rng.choice([rng.randint(1, 99) * 1000, rng.randint(1, 999) * 1000, rng.randint(1, 9999) * 500])
        if big:
            v = rng.randint(10, 9999) * 1000
        s = f"{v:,}"
        style = rng.random()
        if style < 0.45:
            s = s.replace(",", ".")
        elif style < 0.6:
            s = s.replace(",", "")
        if rng.random() < 0.25:
            s += rng.choice(["đ", " đ", " VND", " VNĐ", "d"])
        if rng.random() < 0.1:
            s += ".00"
        return self._ok(s)

    def date(self, rng: random.Random) -> str:
        d, m, y = rng.randint(1, 28), rng.randint(1, 12), rng.randint(2018, 2026)
        sep = rng.choice(["/", "-", "."])
        s = f"{d:02d}{sep}{m:02d}{sep}{y}" if rng.random() < 0.85 else f"{d:02d}{sep}{m:02d}{sep}{y % 100:02d}"
        if rng.random() < 0.7:
            s += f" {rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}"
            if rng.random() < 0.3:
                s += f":{rng.randint(0, 59):02d}"
        return s

    def phone(self, rng: random.Random) -> str:
        digits = "".join(str(rng.randint(0, 9)) for _ in range(rng.choice([8, 9])))
        p = rng.choice(["0", "024", "028", "090", "091", "097", "1900"]) + digits
        return p[: rng.choice([10, 11, 11, 12])]

    def item_name(self, rng: random.Random) -> str:
        if rng.random() < 0.6:
            return rng.choice(self.items)
        n = rng.randint(2, 4)
        s = " ".join(rng.choice(self.words) for _ in range(n))
        if rng.random() < 0.4:
            s += " " + rng.choice(["500g", "1kg", "330ml", "1L", "x2", "loại 1", "2kg"])
        return self._ok(s)

    def seller(self, rng: random.Random) -> str:
        if rng.random() < 0.6:
            return rng.choice(self.sellers)
        s = rng.choice(_SELLER_WORDS) + " " + " ".join(rng.choice(self.words) for _ in range(rng.randint(1, 2)))
        return self._ok(s.upper() if rng.random() < 0.6 else s.title())

    def address(self, rng: random.Random) -> str:
        if rng.random() < 0.5:
            return rng.choice(self.addresses)
        s = f"{rng.choice(['Số', 'ĐC:', '', 'Đ/c:'])} {rng.randint(1, 999)} {rng.choice(_STREETS)}, {rng.choice(['P.', 'Phường', 'Q.', 'Quận'])} {rng.randint(1, 12)}, {rng.choice(_CITIES)}"
        return self._ok(re.sub(r"\s+", " ", s).strip())

    # --- public API -----------------------------------------------------------------------------

    def line(self, rng: random.Random) -> str:
        for _ in range(_MAX_REROLL):
            s = self._line(rng)
            if not self.blocked(s):
                return s
        raise RuntimeError("blocklist rejects every generated line")

    def receipt(self, rng: random.Random) -> list[tuple[str, str]]:
        """Lines of one receipt, top to bottom, as (text, label). Always >= 8 lines."""
        for _ in range(_MAX_REROLL):  # re-draw the whole receipt so totals stay consistent
            rows = self._receipt(rng)
            if not any(self.blocked(t) for t, _ in rows):
                return rows
        raise RuntimeError("blocklist rejects every generated receipt")

    def _line(self, rng: random.Random) -> str:
        r = rng.random()
        if r < 0.18:
            s = self.item_name(rng)
        elif r < 0.30:
            s = self.amount(rng, big=rng.random() < 0.4)
        elif r < 0.40:
            s = rng.choice(["Ngày: ", "Ngày bán: ", "Date: ", "Thời gian: ", ""]) + self.date(rng)
        elif r < 0.48:
            s = rng.choice(["ĐT: ", "Tel: ", "Hotline: ", "SĐT: "]) + self.phone(rng)
        elif r < 0.56:
            s = self.address(rng)
        elif r < 0.63:
            s = self.seller(rng)
        elif r < 0.72:
            s = rng.choice(["Tổng cộng: ", "Tổng tiền: ", "Thành tiền: ", "Tiền mặt: ", "Tiền thừa: ", "TỔNG: "]) + self.amount(rng, big=True)
        elif r < 0.80:
            s = rng.choice(_TITLES + _THANKS)
        elif r < 0.88:
            s = f"{self.item_name(rng)} x{rng.randint(1, 9)}"
        else:  # random word salad for diversity (spec: ~30% mixes words.txt)
            s = " ".join(rng.choice(self.words) for _ in range(rng.randint(1, 5)))
        s = self._ok(s).strip()
        return s if s else "0"

    def _receipt(self, rng: random.Random) -> list[tuple[str, str]]:
        rows: list[tuple[str, str]] = [(self.seller(rng), "SELLER")]
        for _ in range(rng.randint(1, 2)):
            rows.append((self.address(rng), "ADDRESS"))
        if rng.random() < 0.7:
            rows.append((rng.choice(["ĐT: ", "Tel: ", "Hotline: ", "SĐT: "]) + self.phone(rng), "OTHER"))
        rows.append((rng.choice(_TITLES), "OTHER"))
        if rng.random() < 0.5:
            rows.append((f"Số HĐ: {rng.randint(1000, 99999)}  Quầy: {rng.randint(1, 9):02d}", "OTHER"))
        rows.append((rng.choice(["Ngày: ", "Ngày bán: ", "Date: ", "Giờ vào: ", "Ngày "]) + self.date(rng), "TIMESTAMP"))
        if rng.random() < 0.5:
            rows.append((f"Thu ngân: {rng.choice(self.words).title()} {rng.choice(self.words).title()}", "OTHER"))
        rows.append((rng.choice(["Mặt hàng   SL   Đơn giá   T.Tiền", "Tên hàng  SL  Đ.Giá  Thành tiền", "Sản phẩm      SL     Thành tiền"]), "OTHER"))
        total = 0
        for _ in range(rng.randint(2, 7)):
            name, qty = self.item_name(rng), rng.randint(1, 5)
            price = rng.randint(2, 400) * 500
            total += qty * price
            if rng.random() < 0.5:
                rows.append((name, "OTHER"))
                rows.append((f"{qty} x {price:,}".replace(",", ".") + f"   {qty * price:,}".replace(",", "."), "OTHER"))
            else:
                rows.append((f"{name[:22]:<22} {qty:>2} {qty * price:>9,}".replace(",", "."), "OTHER"))
        if rng.random() < 0.6:
            rows.append((rng.choice(["Tổng tiền hàng: ", "Thành tiền: ", "Subtotal: "]) + f"{total:,}".replace(",", "."), "OTHER"))
        if rng.random() < 0.4:
            rows.append((rng.choice(["Chiết khấu: ", "Giảm giá: ", "VAT 8%: "]) + f"{rng.randint(0, 50) * 1000:,}".replace(",", "."), "OTHER"))
        total_text = f"{total:,}".replace(",", ".") if rng.random() < 0.7 else f"{total:,}"
        if rng.random() < 0.5:
            rows.append((rng.choice(["Tổng cộng: ", "TỔNG TIỀN PHẢI T.TOÁN: ", "Tổng tiền: ", "TỔNG CỘNG: ", "Tổng thanh toán: "]) + total_text, "TOTAL_COST"))
        else:  # label on one line, amount on the next (as in MC-OCR)
            rows.append((rng.choice(["Tổng cộng:", "TỔNG TIỀN PHẢI T.TOÁN", "Tổng tiền:", "TỔNG CỘNG"]), "TOTAL_COST"))
            rows.append((total_text, "TOTAL_COST"))
        if rng.random() < 0.7:
            paid = (total // 1000 + rng.randint(0, 100)) * 1000
            rows.append((rng.choice(["Tiền khách trả: ", "Tiền mặt: ", "Khách đưa: "]) + f"{paid:,}".replace(",", "."), "OTHER"))
            rows.append((rng.choice(["Tiền thừa: ", "Tiền trả lại: "]) + f"{max(0, paid - total):,}".replace(",", "."), "OTHER"))
        for _ in range(rng.randint(1, 3)):
            rows.append((rng.choice(_THANKS), "OTHER"))
        out = [(self._ok(t).strip(), lab) for t, lab in rows]
        return [(t if t else "-", lab) for t, lab in out]


class LineRenderer:
    """Render one text line to an HxW RGB uint8 image (H fixed, W <= max_width)."""

    def __init__(self, fonts: list[str], height: int = 32, max_width: int = 320, seed: int = 0):
        if not fonts:
            raise ValueError("need at least one font")
        self.fonts, self.height, self.max_width = list(fonts), height, max_width
        self._cache: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}

    def _font(self, path: str, size: int) -> ImageFont.FreeTypeFont:
        key = (path, size)
        if key not in self._cache:
            self._cache[key] = ImageFont.truetype(path, size)
        return self._cache[key]

    def render(self, text: str, rng: random.Random) -> np.ndarray:
        text = text if text.strip() else " "
        font = self._font(rng.choice(self.fonts), rng.randint(22, 40))
        left, top, right, bottom = font.getbbox(text)
        tw, th = max(1, right - left), max(1, bottom - top)
        pad_x, pad_y = rng.randint(2, 12), rng.randint(1, 6)
        W, H = tw + 2 * pad_x, th + 2 * pad_y

        bg = rng.randint(170, 255)
        bg_rgb = (bg, max(0, bg - rng.randint(0, 20)), max(0, bg - rng.randint(0, 25)))
        ink = rng.randint(0, 90)
        ink_rgb = (ink, ink + rng.randint(0, 20), ink + rng.randint(0, 40))
        img = Image.new("RGB", (W, H), bg_rgb)
        ImageDraw.Draw(img).text((pad_x - left, pad_y - top), text, font=font, fill=ink_rgb)
        arr = np.asarray(img).astype(np.float32)

        # photo-like degradations (synthtiger postprocess: noise, gaussian/motion blur, contrast)
        if rng.random() < 0.5:
            sigma = rng.uniform(1, 8)
            arr += np.random.default_rng(rng.getrandbits(32)).normal(0, sigma, (H, W)).astype(np.float32)[..., None]
        if rng.random() < 0.3:
            k = rng.choice([3, 5])
            arr = cv2.GaussianBlur(arr, (k, k), 0)
        if rng.random() < 0.2:
            arr = arr * rng.uniform(0.8, 1.2) + rng.uniform(-20, 20)
        arr = np.clip(arr, 0, 255).astype(np.uint8)

        # small rotation (crops are already rectified by crop_for_rec, so keep it <= 3°)
        if rng.random() < 0.4:
            M = cv2.getRotationMatrix2D((W / 2, H / 2), rng.uniform(-3, 3), 1.0)
            arr = cv2.warpAffine(arr, M, (W, H), borderMode=cv2.BORDER_REPLICATE)

        scale = self.height / H
        new_w = max(1, int(round(W * scale)))
        out = cv2.resize(arr, (min(new_w, self.max_width), self.height), interpolation=cv2.INTER_AREA)
        return out
