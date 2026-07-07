"""
date_extractor.py — 從 OCR 文字中擷取並解析日期資訊

支援格式：yyyy.mm.dd  |  yyyy/mm/dd  |  yyyy mm dd

公開 API：
  extract_date_str(text)  → 原始日期字串 or None
  parse_date(date_str)    → {"year":str, "month":str, "day":str} or None
  extract_date_type(text) → "expiry" | "manufacture"
  extract_dates(text)     → {"expiry": dict|None, "manufacture": dict|None}
"""

import re
from datetime import date
from typing import Optional

from dateutil.relativedelta import relativedelta

# ── 日期 Regex（依優先順序：點 > 斜線 > 空白 > 無分隔）──────────────────────
_PAT_DOT     = re.compile(r'(\d{4})\.(\d{1,2})\.(\d{1,2})')
_PAT_SLASH   = re.compile(r'(\d{4})/(\d{1,2})/(\d{1,2})')
_PAT_SPACE   = re.compile(r'(\d{4})[ \t]+(\d{1,2})[ \t]+(\d{1,2})')
# yyyymmdd 無分隔：month 限 01-12、day 限 01-31，且前後不可緊鄰其他數字（避免誤中條碼）
_PAT_COMPACT = re.compile(
    r'(?<!\d)'
    r'(\d{4})'
    r'(0[1-9]|1[0-2])'
    r'(0[1-9]|[12]\d|3[01])'
    r'(?!\d)'
)

_PATTERNS = [_PAT_DOT, _PAT_SLASH, _PAT_SPACE, _PAT_COMPACT]

# 關鍵字（較長的優先，避免「製造日期」被「製造」截斷）
_MANUFACTURE_KW = ["製造日期", "製造"]
_EXPIRY_KW      = ["有效日期", "有效期限", "有效期"]

# 保存期限：匹配「保存期限」後接數字與「個月」，中間允許任意分隔符與空白
_PAT_SHELF = re.compile(r'保存期限[：: \t]*(\d+)\s*個月')


# ── 公開函式 ──────────────────────────────────────────────────────────────────

def extract_date_str(text: str) -> Optional[str]:
    """
    從任意 OCR 字串中提取第一個符合格式的日期子字串。
    日期前後可有任意字元。

    >>> extract_date_str("有效日期:2026 10 10 07:56 002")
    '2026 10 10'
    >>> extract_date_str("2027.01.13.19:55\\n樂事")
    '2027.01.13'
    """
    for pat in _PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(0)
    return None


def parse_date(date_str: str) -> Optional[dict]:
    """
    將日期字串解析為結構化 dict。

    >>> parse_date("2027.01.13")
    {'year': '2027', 'month': '01', 'day': '13'}
    >>> parse_date("2026/5/5")
    {'year': '2026', 'month': '05', 'day': '05'}
    """
    for pat in _PATTERNS:
        m = pat.search(date_str)
        if m:
            return {
                "year":  m.group(1),
                "month": m.group(2).zfill(2),
                "day":   m.group(3).zfill(2),
            }
    return None


def extract_date_type(text: str) -> str:
    """
    根據文字中的關鍵字判斷日期類型。

    規則：
      - 含「製造」相關字 → "manufacture"
      - 含「有效日期」相關字，或無任何標籤 → "expiry"（預設）

    >>> extract_date_type("製造2026/05/05")
    'manufacture'
    >>> extract_date_type("有效日期:2027.01.13")
    'expiry'
    >>> extract_date_type("2027.01.13")
    'expiry'
    """
    for kw in _MANUFACTURE_KW:
        if kw in text:
            return "manufacture"
    return "expiry"


def extract_dates(text: str) -> dict:
    """
    從 OCR 文字中同時擷取有效日期（expiry）與製造日期（manufacture）。

    規則：
      1. 找「製造」系列關鍵字後的日期 → manufacture
      2. 找「有效日期」系列關鍵字後的日期 → expiry
      3. 若無 expiry 標籤，第一個出現的日期預設為 expiry

    回傳：
      {"expiry": {"year":str, "month":str, "day":str} | None,
       "manufacture": {"year":str, "month":str, "day":str} | None}

    >>> extract_dates("製造2026/05/05\\n有效日期:2027/05/05")
    {'expiry': {'year': '2027', 'month': '05', 'day': '05'}, 'manufacture': {'year': '2026', 'month': '05', 'day': '05'}}
    >>> extract_dates("2027.01.13.19:55\\n樂事洋芋片")
    {'expiry': {'year': '2027', 'month': '01', 'day': '13'}, 'manufacture': None}
    """
    result: dict = {"expiry": None, "manufacture": None}

    # 步驟 1：尋找製造日期
    for kw in _MANUFACTURE_KW:
        d = _date_after_keyword(text, kw)
        if d:
            result["manufacture"] = d
            break

    # 步驟 2：若有製造日期，嘗試從「保存期限 X 個月」推算有效日期
    if result["manufacture"] is not None:
        months = extract_shelf_life_months(text)
        if months is not None:
            result["expiry"] = _add_months(result["manufacture"], months)
            result["shelf_life_months"] = months

    # 步驟 3：尋找明確標示的有效日期（優先覆蓋推算值）
    for kw in _EXPIRY_KW:
        d = _date_after_keyword(text, kw)
        if d:
            result["expiry"] = d
            break

    # 步驟 4：仍無有效日期時，第一個出現的日期預設為 expiry
    if result["expiry"] is None:
        date_str = extract_date_str(text)
        if date_str:
            parsed = parse_date(date_str)
            if parsed and parsed != result["manufacture"]:
                result["expiry"] = parsed

    return result


def extract_shelf_life_months(text: str) -> Optional[int]:
    """
    從文字中擷取「保存期限 X 個月」的 X 值。

    >>> extract_shelf_life_months("保存期限:10個月")
    10
    >>> extract_shelf_life_months("保存期限 6 個月")
    6
    >>> extract_shelf_life_months("無相關資訊")
    """
    m = _PAT_SHELF.search(text)
    return int(m.group(1)) if m else None


# ── 內部輔助 ──────────────────────────────────────────────────────────────────

def _date_after_keyword(text: str, keyword: str) -> Optional[dict]:
    """在關鍵字之後的文字中提取第一個日期。"""
    idx = text.find(keyword)
    if idx == -1:
        return None
    after = text[idx + len(keyword):]
    date_str = extract_date_str(after)
    return parse_date(date_str) if date_str else None


def _add_months(d: dict, months: int) -> dict:
    """將日期往後推算指定月數，閏年與月底溢位由 relativedelta 處理。"""
    dt = date(int(d["year"]), int(d["month"]), int(d["day"]))
    dt2 = dt + relativedelta(months=months)
    return {
        "year":  str(dt2.year),
        "month": str(dt2.month).zfill(2),
        "day":   str(dt2.day).zfill(2),
    }


# ── 快速自測 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cases = [
        "2027.01.13.19:55\n樂事\n洋芋片\n青檸享清新口味",
        "製造2026/05/05\n50Gx10入\n玉黍叔\n甜辣口味",
        "製造2026/05/05\n保存期限:10個月\n玉黍叔",          # 推算有效日期
        "製造2026/07/31\n保存期限:1個月\n月底溢位測試",      # 7/31 + 1月 = 8/31
        "有效日期：\n280g(14人)X12包\n2027.03.15 65859\n益生菌",
        "有效日期:2026 10 10 07:56 002 00232 1B2\n科学超",
        "有效日期:2026.10.05 07:15 002\n韓式泡菜\n來一客",
        "2027 10 09 C072\n維他露P\n易開罐250毫升x24罐",
        "製造2026/05/04\n有效日期:2027/05/04\n華元真魷味",   # 明確有效日期優先
        "華元\n製造2026/05/05\n50GX10入\n真魷味\n保存期限:10個月",
        "MOS BURGER\n6包/盒；4盒/箱\n有效日期：20280409",     # yyyymmdd 無分隔
        "barcode 14710421020084\n有效日期：20280409",         # 條碼不誤判
    ]

    for ocr in cases:
        first_line = ocr.split("\n")[0]
        result = extract_dates(ocr)
        print(f"OCR : {first_line}")
        print(f"  expiry   : {result['expiry']}")
        print(f"  manufacture: {result['manufacture']}")
        print()
