"""
個人情報 自動マスキングエンジン（完全ローカル処理）

流れ:  動画のコマを読む → OCR(PaddleOCRのモデル / RapidOCR)で文字を読む
       → 個人情報っぽい文字を判定 → その部分だけ塗りつぶし/モザイク → 動画に戻す
※ 処理はすべてこのパソコンの中。動画も文字もネットには送りません。
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import unicodedata
from dataclasses import dataclass, field

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 1. OCR（PaddleOCR PP-OCRv5 モデルを RapidOCR で実行。日本語・英語対応）
# ---------------------------------------------------------------------------

_engine = None
_engine_lock = threading.RLock()


def _get_engine():
    global _engine
    with _engine_lock:
        if _engine is None:
            from rapidocr import ModelType, OCRVersion, RapidOCR
            _engine = RapidOCR(params={
                "Global.log_level": "error",
                "Det.ocr_version": OCRVersion.PPOCRV5, "Rec.ocr_version": OCRVersion.PPOCRV5,
                "Det.model_type": ModelType.MOBILE, "Rec.model_type": ModelType.MOBILE,
            })
        return _engine


def _rect(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))


@dataclass
class Line:
    text: str
    box: tuple  # (x1, y1, x2, y2) ピクセル
    spans: list = field(default_factory=list, repr=False)  # [(開始, 終了, 枠)] 文字ごとの位置

    def sub_box(self, start: int, end: int):
        """行の中の一部分（start〜end 文字目）だけの枠を返す"""
        if start <= 0 and end >= len(self.text):
            return self.box
        x1, y1, x2, y2 = self.box
        boxes = []
        for s, e, (bx1, by1, bx2, by2) in self.spans:
            if e <= start or s >= end:
                continue
            n = max(1, e - s)  # 単語の途中で切れる場合は幅を按分
            fs = (max(s, start) - s) / n
            fe = (min(e, end) - s) / n
            w = bx2 - bx1
            boxes.append((int(bx1 + w * fs), by1, int(bx1 + w * fe + 0.5), by2))
        if not boxes:  # 位置情報が取れなければ文字数で按分
            n = max(1, len(self.text))
            w = x2 - x1
            return (int(x1 + w * start / n), y1, int(x1 + w * end / n + 0.5), y2)
        return (min(b[0] for b in boxes), min(b[1] for b in boxes),
                max(b[2] for b in boxes), max(b[3] for b in boxes))


def ocr(frame_bgr: np.ndarray) -> list[Line]:
    with _engine_lock:
        res = _get_engine()(frame_bgr, return_word_box=True)
    if res is None or res.txts is None:
        return []
    words_all = res.word_results or [()] * len(res.txts)
    lines = []
    for box, text, words in zip(res.boxes, res.txts, words_all):
        text = str(text)
        spans, pos = [], 0
        for w in words or ():
            tok = str(w[0])
            p = text.find(tok, pos)
            if p < 0:
                continue
            spans.append((p, p + len(tok), _rect(w[2])))
            pos = p + len(tok)
        lines.append(Line(text, _rect(box), spans))
    return lines


# ---------------------------------------------------------------------------
# 2. 個人情報の判定ルール
# ---------------------------------------------------------------------------

DASH = r"[-‐‑‒–—―ー−－]"
PREFS = (
    "北海道|青森県|岩手県|宮城県|秋田県|山形県|福島県|茨城県|栃木県|群馬県|埼玉県|千葉県|東京都|"
    "神奈川県|新潟県|富山県|石川県|福井県|山梨県|長野県|岐阜県|静岡県|愛知県|三重県|滋賀県|京都府|"
    "大阪府|兵庫県|奈良県|和歌山県|鳥取県|島根県|岡山県|広島県|山口県|徳島県|香川県|愛媛県|高知県|"
    "福岡県|佐賀県|長崎県|熊本県|大分県|宮崎県|鹿児島県|沖縄県"
)

# (カテゴリ名, 正規表現, マスクする部分のグループ番号)
PATTERNS = {
    "email": [("メールアドレス", r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]*", 0)],
    "phone": [
        ("電話番号", rf"(?<!\d)(?:\+81{DASH}?|0)\d{{1,4}}{DASH}?\(?\d{{1,4}}\)?{DASH}?\d{{3,4}}(?!\d)", 0),
        ("電話番号", r"(?<!\d)0[5789]0\d{8}(?!\d)", 0),
    ],
    "postal": [("郵便番号", rf"〒\s*\d{{3}}{DASH}?\d{{4}}|(?<![\d\-])\d{{3}}{DASH}\d{{4}}(?![\d\-])", 0)],
    "card": [
        ("カード/マイナンバー等の番号", r"(?<!\d)\d{4}[ \-]?\d{4}[ \-]?\d{4}(?:[ \-]?\d{4})?(?!\d)", 0),
    ],
    "address": [
        ("住所", rf"(?:{PREFS}).{{2,}}", 0),
        ("住所", rf"\S{{1,8}}[市区町村郡]\S*?\d+(?:丁目|番地?|号|{DASH}\d+)\S*", 0),
    ],
    "honorific": [
        # 「山田太郎様」「田中さん」の名前部分
        ("敬称つきの名前", r"([一-龥々〆ァ-ヶー]{1,6}[ 　]?[一-龥々〆ァ-ヶー]{0,6})[ 　]?(?:様|さま|さん|殿|御中)", 1),
    ],
}

# この見出しの「後ろ」や「右・下の欄」を隠す
LABELS = [
    "氏名", "お名前", "名前", "フリガナ", "ふりがな", "カナ", "姓", "名",
    "住所", "ご住所", "所在地", "電話番号", "電話", "TEL", "携帯", "メールアドレス", "メール",
    "E-mail", "Email", "生年月日", "誕生日", "口座番号", "カード番号", "会員番号", "顧客番号",
    "担当者", "ユーザー名", "アカウント名", "宛名", "差出人", "送信者", "宛先",
    "パスワード", "暗証番号", "ログインID", "備考", "備考欄", "会員情報", "会員名", "顧客名", "顧客情報", "お客様名", "お客様情報",
    "申込者", "申込者名", "購入者", "購入者名", "契約者", "契約者名", "ニックネーム", "LINE名",
]
# 表の列見出しとして使われるもの（この下の列をまとめて隠す）
COLUMN_HEADERS = {"氏名", "お名前", "名前", "会員情報", "会員名", "顧客名", "顧客情報", "お客様名", "お客様情報",
                  "申込者", "申込者名", "購入者", "購入者名", "契約者", "契約者名", "メールアドレス", "メール",
                  "電話番号", "住所", "ユーザー名", "アカウント名", "ニックネーム", "LINE名", "担当者", "宛先", "差出人", "送信者"}
LABEL_RE = re.compile(
    r"^\s*[【\[（(]?\s*(" + "|".join(sorted(map(re.escape, LABELS), key=len, reverse=True)) + r")\s*[*＊※]?\s*[】\]）)]?\s*(?:[:：]\s*|\s+|$)",
    re.IGNORECASE,
)


# OCRが日本語の漢字を中国の簡体字で読むことがあるので直す（例: 会员情報 → 会員情報）
SIMPLIFIED = str.maketrans({"员": "員", "话": "話", "电": "電", "邮": "郵", "码": "碼", "账": "帳",
                            "户": "戸", "机": "機", "号": "号", "称": "称", "务": "務", "关": "関",
                            "联": "聯", "络": "絡", "约": "約", "购": "購", "买": "買", "顾": "顧",
                            "详": "詳", "细": "細", "设": "設", "页": "頁", "姓": "姓", "现": "現"})


def _normalize(text: str):
    """全角→半角などを揃え、元の文字位置との対応表も作る"""
    out, idx = [], []
    for i, ch in enumerate(text):
        n = unicodedata.normalize("NFKC", ch).translate(SIMPLIFIED)
        for c in n:
            out.append(c)
            idx.append(i)
    return "".join(out), idx


HONORIFIC_STOP = {"客", "お客", "皆", "皆々", "各位", "担当", "ご担当", "担当者", "会員", "ユーザー",
                  "利用者", "ご利用者", "お子", "神", "仏", "王", "奥", "旦那", "先方", "相手", "みな"}


def _is_common_word(with_prev: str, name: str) -> bool:
    name = name.strip()
    return name in HONORIFIC_STOP or with_prev.strip() in HONORIFIC_STOP or name.endswith(("客", "者", "員"))


DATE_RE = re.compile(r"(?:19|20)\d{2}\s*[-/.年]\s*\d{1,2}\s*[-/.月]\s*\d{1,2}|\d{1,2}\s*:\s*\d{2}")


def _looks_like_datetime(norm: str, s: int, e: int) -> bool:
    """2025-04-12 12:07:39 のような日時を電話番号などと間違えない"""
    for m in DATE_RE.finditer(norm):
        if m.start() < e and m.end() > s:
            return True
    return False


def _span_back(idx, s, e, n_orig):
    if s >= len(idx):
        return n_orig, n_orig
    a = idx[s]
    b = idx[e - 1] + 1 if e - 1 < len(idx) else n_orig
    return a, b


@dataclass
class Hit:
    category: str
    text: str
    box: tuple


def _custom_hits(line: Line, words: list[str]) -> list[Hit]:
    """登録ワード（スペースや全角半角の違いは無視して一致）"""
    norm, idx = _normalize(line.text)
    # 空白を抜いた文字列と位置対応
    chars = [(c, i) for c, i in zip(norm, idx) if not c.isspace()]
    flat = "".join(c for c, _ in chars).lower()
    hits = []
    for w in words:
        wn = re.sub(r"\s+", "", unicodedata.normalize("NFKC", w)).lower()
        if not wn:
            continue
        start = 0
        while True:
            p = flat.find(wn, start)
            if p < 0:
                break
            a = chars[p][1]
            b = chars[p + len(wn) - 1][1] + 1
            hits.append(Hit("登録ワード", line.text[a:b], line.sub_box(a, b)))
            start = p + len(wn)
    return hits


def detect(lines: list[Line], cats: set[str], words: list[str]) -> list[Hit]:
    hits: list[Hit] = []
    label_only: list[Line] = []

    for ln in lines:
        norm, idx = _normalize(ln.text)
        for cat in cats:
            for name, pat, grp in PATTERNS.get(cat, []):
                for m in re.finditer(pat, norm):
                    s, e = m.span(grp)
                    if e <= s:
                        continue
                    if cat == "honorific" and _is_common_word(norm[max(0, s - 1):e], norm[s:e]):
                        continue
                    if cat in ("phone", "postal", "card") and _looks_like_datetime(norm, s, e):
                        continue
                    a, b = _span_back(idx, s, e, len(ln.text))
                    hits.append(Hit(name, ln.text[a:b], ln.sub_box(a, b)))

        if "label" in cats:
            m = LABEL_RE.match(norm)
            if m:
                rest = norm[m.end():].strip()
                if rest:  # 「氏名：山田太郎」→ 後ろを隠す
                    a, _ = _span_back(idx, m.end(), m.end() + 1, len(ln.text))
                    hits.append(Hit(f"「{m.group(1)}」欄", ln.text[a:], ln.sub_box(a, len(ln.text))))
                else:  # 見出しだけの行 → 右か下の欄を探す
                    label_only.append(ln)

        if words:
            hits.extend(_custom_hits(ln, words))

    # 見出しだけの行のまわりを隠す
    #  ・表の列見出し（下に同じ列の行が続く）→ その列を下までまとめて隠す
    #  ・フォームの見出し → 同じ行の右側の欄（なければ真下の欄）を隠す
    def is_label(ln):
        return bool(LABEL_RE.match(_normalize(ln.text)[0]))

    for lab in label_only:
        lx1, ly1, lx2, ly2 = lab.box
        lh = max(ly2 - ly1, 8)
        name = LABEL_RE.match(_normalize(lab.text)[0]).group(1)

        # 表の列: 見出しと左端がそろって、すき間なく下に並ぶ行
        col, bottom = [], ly2
        for ln in sorted((l for l in lines if l.box[1] >= ly2 - 2), key=lambda l: l.box[1]):
            x1, y1, x2, y2 = ln.box
            if abs(x1 - lx1) > lh * 1.5 or ln is lab:
                continue
            if y1 - bottom > lh * 2.5 or is_label(ln):
                break
            col.append(ln)
            bottom = y2
        # 表の見出し行なら、同じ高さの右側に他の見出しが3つ以上並んでいる
        row_mates = sum(1 for l in lines if l is not lab and l.box[0] > lx2
                        and abs((l.box[1] + l.box[3]) / 2 - (ly1 + ly2) / 2) <= lh * 0.6)
        if name in COLUMN_HEADERS and len(col) >= 2 and row_mates >= 3:
            hits.extend(Hit(f"「{name}」列", ln.text, ln.box) for ln in col)
            continue

        # フォーム: 同じ行（見出しの高さ±1行）で右にある欄
        cy = (ly1 + ly2) / 2
        right = [ln for ln in lines if ln is not lab
                 and ln.box[0] >= lx2 - 4 and ln.box[0] - lx2 < lh * 12
                 and abs((ln.box[1] + ln.box[3]) / 2 - cy) <= lh * 0.9
                 and not is_label(ln)]
        if name in ("パスワード", "暗証番号"):
            # 入力中のパスワードは選択色などで文字として読めないことがあるので、行の右側を丸ごと隠す
            xs = [ln.box[2] for ln in right] + [lx2 + lh * 20]
            ys1 = [ln.box[1] for ln in right] + [ly1 - lh // 2]
            ys2 = [ln.box[3] for ln in right] + [ly2 + lh // 2]
            hits.append(Hit(f"「{name}」欄", "（パスワード欄）", (lx2 + lh // 2, min(ys1), max(xs), max(ys2))))
            continue
        if right:
            hits.extend(Hit(f"「{name}」欄", ln.text, ln.box) for ln in right)
            continue
        below = [ln for ln in lines if ln is not lab
                 and min(lx2, ln.box[2]) - max(lx1, ln.box[0]) > 0
                 and 0 <= ln.box[1] - ly2 + 2 < lh * 1.5]
        if below and not is_label(below[0]):
            hits.append(Hit(f"「{name}」欄", below[0].text, below[0].box))

    return hits


# ---------------------------------------------------------------------------
# 3. 塗りつぶし / モザイク
# ---------------------------------------------------------------------------


def apply_masks(frame, boxes, style="fill", pad=4):
    h, w = frame.shape[:2]
    for (x1, y1, x2, y2) in boxes:
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
        if x2 <= x1 or y2 <= y1:
            continue
        if style == "mosaic":
            roi = frame[y1:y2, x1:x2]
            bs = max(6, (y2 - y1) // 2)  # 文字が読めない粗さ
            small = cv2.resize(roi, (max(1, (x2 - x1) // bs), max(1, (y2 - y1) // bs)), interpolation=cv2.INTER_AREA)
            frame[y1:y2, x1:x2] = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
        else:
            cv2.rectangle(frame, (x1, y1), (x2 - 1, y2 - 1), (60, 60, 60), -1)
    return frame


def _hide(text: str) -> str:
    """結果一覧に出すとき用（中身が分からない程度に伏せる）"""
    t = text.strip()
    if len(t) <= 2:
        return "＊" * len(t)
    return t[0] + "＊" * (len(t) - 2) + t[-1]


# ---------------------------------------------------------------------------
# 4. 画像 / 動画の処理
# ---------------------------------------------------------------------------

DEFAULT_CATS = {"email", "phone", "postal", "card", "address", "honorific", "label"}


def process_image(src, dst, cats=DEFAULT_CATS, words=(), style="fill", fixed_boxes=(), **_):
    img = cv2.imread(src)
    if img is None:
        raise ValueError("画像を読み込めませんでした")
    hits = detect(ocr(img), set(cats), list(words))
    apply_masks(img, [h.box for h in hits] + list(fixed_boxes), style)
    cv2.imwrite(dst, img)
    return [{"time": 0, "category": h.category, "text": _hide(h.text)} for h in hits]


def process_video(src, dst, cats=DEFAULT_CATS, words=(), style="fill", interval=0.5,
                  fixed_boxes=(), progress=lambda p, msg: None):
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise ValueError("動画を読み込めませんでした")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = max(1, int(round(fps * interval)))
    min_gap = max(1, int(fps * 0.15))

    # --- 1周目: 文字を読む（一定間隔 ＋ 画面が大きく変わった瞬間） ---
    ocr_lines: dict[int, list[Line]] = {}
    prev_small, last = None, -10**9
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 90))
        changed = prev_small is not None and float(np.mean(cv2.absdiff(small, prev_small))) > 6
        if i - last >= step or (changed and i - last >= min_gap) or i == 0:
            ocr_lines[i] = ocr(frame)
            last = i
            prev_small = small
        elif prev_small is None:
            prev_small = small
        if i % 10 == 0:
            progress(min(0.8, 0.8 * i / total), f"文字を読み取り中… {i}/{total}コマ")
        i += 1
    total = i
    cap.release()

    # 一度でも個人情報と判定された文字は「覚えて」、動画中のどこに出ても隠す
    # （スクロールで見出しが画面外に出た後の名前、なども取りこぼさない）
    progress(0.8, "判定中…")
    first = {f: detect(ls, set(cats), list(words)) for f, ls in ocr_lines.items()}
    learned = {re.sub(r"\s+", "", h.text) for hs in first.values() for h in hs}
    learned = [t for t in learned if len(t) >= 2]
    ocr_frames: dict[int, list[Hit]] = {}
    for f, ls in ocr_lines.items():
        extra = [h for ln in ls for h in _custom_hits(ln, learned)]
        for h in extra:
            h.category = "同じ文字（他の場面で検出）"
        ocr_frames[f] = first[f] + extra
    ocr_frames.setdefault(total, [])  # 番兵

    # --- 2周目: 前後の読み取り結果を合わせて塗る（取りこぼし防止） ---
    keys = sorted(ocr_frames)
    tmp_dir = tempfile.mkdtemp()
    tmp_video = os.path.join(tmp_dir, "v.mp4")
    writer = cv2.VideoWriter(tmp_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    cap = cv2.VideoCapture(src)
    k = 0
    for i in range(total):
        ok, frame = cap.read()
        if not ok:
            break
        while k + 1 < len(keys) and keys[k + 1] <= i:
            k += 1
        boxes = [h.box for h in ocr_frames[keys[k]]]
        if k + 1 < len(keys):
            boxes += [h.box for h in ocr_frames[keys[k + 1]]]
        if k - 1 >= 0 and i - keys[k] < min_gap:  # 切り替わり直後は直前も
            boxes += [h.box for h in ocr_frames[keys[k - 1]]]
        writer.write(apply_masks(frame, boxes + list(fixed_boxes), style))
        if i % 15 == 0:
            progress(0.8 + 0.15 * i / total, f"塗りつぶし中… {i}/{total}コマ")
    writer.release()
    cap.release()

    # --- 音声を戻して、どこでも再生できる形式(H.264)で保存 ---
    progress(0.96, "動画を書き出し中…")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", tmp_video, "-i", src,
               "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "libx264", "-pix_fmt", "yuv420p",
               "-crf", "20", "-preset", "veryfast", "-c:a", "aac", "-shortest", dst]
        subprocess.run(cmd, check=True)
    else:
        shutil.move(tmp_video, dst)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # 結果一覧（同じ内容は最初に出た時刻だけ）
    seen, report = set(), []
    for f in keys:
        for h in sorted(ocr_frames[f], key=lambda h: h.category.startswith("同じ文字")):
            key = re.sub(r"\s+", "", h.text)
            if key in seen:
                continue
            seen.add(key)
            report.append({"time": round(f / fps, 1), "category": h.category, "text": _hide(h.text)})
    progress(1.0, "完了")
    return report


if __name__ == "__main__":
    import sys
    src, dst = sys.argv[1], sys.argv[2]
    fn = process_video if src.lower().endswith((".mp4", ".mov", ".m4v", ".webm", ".avi", ".mkv")) else process_image
    rep = fn(src, dst, progress=lambda p, m: print(f"\r{p*100:5.1f}% {m}", end="", flush=True))
    print()
    print(json.dumps(rep, ensure_ascii=False, indent=1))
