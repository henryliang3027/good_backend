"""
evaluate/glm_ocr_accuracy.py — GLM-OCR 日期辨識準確率評測

資料集：evaluate/Date-Real/
  images/       ── 510 張日期裁切圖
  annotations.json ── {filename: {ann: [{cls, bbox, transcription}]}}

前置條件：
  llama-server 已在外部啟動（參考 test_glm_ocr.py）

用法：
  python evaluate/glm_ocr_accuracy.py
  python evaluate/glm_ocr_accuracy.py --limit 50          # 只跑前 50 張
  python evaluate/glm_ocr_accuracy.py --resume            # 跳過已完成項目繼續跑
  LLAMA_CPP_URL=http://localhost:8000/v1 python evaluate/glm_ocr_accuracy.py
"""

import argparse
import base64
import io
import json
import os
import re
import shutil
import time
from pathlib import Path

from openai import OpenAI
from PIL import Image

# ── 路徑設定 ──────────────────────────────────────────────────────────────────
_EVAL_DIR     = Path(__file__).parent / "Date-Real"
IMAGES_DIR    = _EVAL_DIR / "images"
ANN_PATH      = _EVAL_DIR / "annotations.json"
RESULTS_PATH  = _EVAL_DIR / "glm_ocr_results.json"

# ── GLM-OCR 設定 ──────────────────────────────────────────────────────────────
LLAMA_CPP_URL = os.getenv("LLAMA_CPP_URL", "http://localhost:8000/v1")
OCR_PROMPT    = "請對這張圖片進行 OCR，只輸出圖中所有數字與文字，不要額外說明。"

_client = OpenAI(base_url=LLAMA_CPP_URL, api_key="no-key-needed")


# ── GLM-OCR 推論 ──────────────────────────────────────────────────────────────
def _pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def call_glm_ocr(img: Image.Image) -> tuple[str, float]:
    """回傳 (ocr_text, elapsed_sec)。"""
    b64 = _pil_to_b64(img)
    t0 = time.time()
    resp = _client.chat.completions.create(
        model="glm-ocr",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": OCR_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }],
        temperature=0,
        max_tokens=64,
    )
    return resp.choices[0].message.content.strip(), time.time() - t0


# ── 準確率計算 ────────────────────────────────────────────────────────────────
def extract_tokens(text: str) -> set[str]:
    """從 OCR 輸出取出所有連續數字/字母 token。"""
    return set(re.findall(r"[A-Za-z0-9]+", text))


def field_hit(transcription: str, ocr_tokens: set[str]) -> bool:
    """
    判斷 ground truth 是否命中：
    1. transcription 本身是獨立 token
    2. 去除前導零後命中（'05' → '5'）
    3. transcription 是某個長數字 token（≥6位）的子字串
       ── 處理無分隔符融合日期，如 '22062022'（DD/MM/YYYY）
    """
    if transcription in ocr_tokens:
        return True
    stripped = transcription.lstrip("0") or "0"
    if stripped in ocr_tokens:
        return True
    for tok in ocr_tokens:
        if len(tok) >= 6 and tok.isalnum() and transcription in tok:
            return True
    return False


def cer(pred: str, gt: str) -> float:
    """字元錯誤率 = Levenshtein distance / len(gt)。"""
    pred_d = re.sub(r"[^0-9]", "", pred)   # 只保留數字做比對
    gt_d   = re.sub(r"[^0-9]", "", gt)
    if not gt_d:
        return 0.0
    m, n = len(pred_d), len(gt_d)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if pred_d[i-1] == gt_d[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[n] / n


# ── 主流程 ────────────────────────────────────────────────────────────────────
def run(limit: int | None, resume: bool) -> None:
    with open(ANN_PATH, encoding="utf-8") as f:
        annotations: dict = json.load(f)

    # 載入已完成結果（resume 模式）
    results: dict = {}
    if resume and RESULTS_PATH.exists():
        with open(RESULTS_PATH, encoding="utf-8") as f:
            saved = json.load(f)
            results = saved.get("details", {})
        print(f"[resume] 已載入 {len(results)} 筆已完成結果")

    filenames = sorted(annotations.keys())
    if limit:
        filenames = filenames[:limit]

    total = len(filenames)
    print(f"評測圖片數：{total}　伺服器：{LLAMA_CPP_URL}\n")

    for idx, fname in enumerate(filenames, 1):
        if fname in results:
            print(f"[{idx:>4}/{total}] {fname}  ── 略過（已完成）")
            continue

        img_path = IMAGES_DIR / fname
        ann_list = annotations[fname]["ann"]
        gt_by_cls = {a["cls"]: a["transcription"] for a in ann_list}

        try:
            img = Image.open(img_path).convert("RGB")
            ocr_text, elapsed = call_glm_ocr(img)
        except Exception as e:
            print(f"[{idx:>4}/{total}] {fname}  ── 錯誤：{e}")
            results[fname] = {"error": str(e), "gt": gt_by_cls}
            _save(results)
            continue

        tokens = extract_tokens(ocr_text)

        # 每個欄位是否命中
        hits = {cls: field_hit(gt, tokens) for cls, gt in gt_by_cls.items()}
        all_correct = all(hits.values())

        # 合併 gt 數字字串供 CER 計算
        gt_concat  = "".join(gt_by_cls.get(c, "") for c in ["year", "month", "day"])
        char_err   = cer(ocr_text, gt_concat)

        results[fname] = {
            "gt":          gt_by_cls,
            "ocr":         ocr_text,
            "hits":        hits,
            "all_correct": all_correct,
            "cer":         round(char_err, 4),
            "elapsed":     round(elapsed, 3),
        }

        status = "✓" if all_correct else "✗"
        hit_str = "  ".join(f"{c}={'✓' if h else '✗'}" for c, h in hits.items())
        print(f"[{idx:>4}/{total}] {fname}  {status}  {hit_str}  CER={char_err:.3f}  ({elapsed:.2f}s)")
        print(f"          GT={gt_by_cls}  OCR={ocr_text!r}")

        # 每 10 張存一次中間結果
        if idx % 10 == 0:
            _save(results)

    _save(results)
    _print_summary(results, total)
    export_failed(results)


def _save(results: dict) -> None:
    summary = _compute_summary(results)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": results}, f, ensure_ascii=False, indent=2)


def _compute_summary(results: dict) -> dict:
    completed = [v for v in results.values() if "error" not in v]
    if not completed:
        return {}

    n = len(completed)
    full_acc  = sum(1 for v in completed if v["all_correct"]) / n
    avg_cer   = sum(v["cer"] for v in completed) / n
    avg_time  = sum(v["elapsed"] for v in completed) / n

    # 各欄位準確率
    field_acc: dict[str, float] = {}
    for cls in ["year", "month", "day"]:
        hits = [v["hits"].get(cls, False) for v in completed if cls in v.get("hits", {})]
        field_acc[cls] = sum(hits) / len(hits) if hits else 0.0

    errors = len(results) - n

    return {
        "evaluated":       n,
        "errors":          errors,
        "full_date_acc":   round(full_acc, 4),
        "field_acc":       {k: round(v, 4) for k, v in field_acc.items()},
        "avg_cer":         round(avg_cer, 4),
        "avg_elapsed_sec": round(avg_time, 3),
    }


def export_failed(results: dict) -> None:
    """將失敗案例複製到 failed/ 資料夾，並產生 result.txt。"""
    failed_dir = _EVAL_DIR / "failed"
    failed_dir.mkdir(exist_ok=True)

    failed = {
        fname: v
        for fname, v in results.items()
        if not v.get("all_correct", True)   # error 或 all_correct=False 都算失敗
    }

    if not failed:
        print("\n無失敗案例，不建立 failed/ 資料夾內容。")
        return

    # 清空舊檔
    for f in failed_dir.glob("*.jpg"):
        f.unlink()
    txt_path = failed_dir / "result.txt"

    lines: list[str] = []
    copied = 0
    for fname in sorted(failed.keys()):
        v = failed[fname]
        src = IMAGES_DIR / fname
        if src.exists():
            shutil.copy2(src, failed_dir / fname)
            copied += 1

        gt      = v.get("gt", {})
        ocr     = v.get("ocr", "（推論失敗）")
        hits    = v.get("hits", {})
        err_msg = v.get("error", "")

        gt_str = "  ".join(f"{c}={gt.get(c,'?')}" for c in ["year", "month", "day"])
        if hits:
            hit_str = "  ".join(f"{c}={'✓' if hits.get(c) else '✗'}" for c in ["year", "month", "day"])
        else:
            hit_str = f"error: {err_msg}"

        lines.append(f"[{fname}]")
        lines.append(f"GT : {gt_str}")
        lines.append(f"OCR: {ocr}")
        lines.append(f"    {hit_str}")
        lines.append("")

    txt_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n失敗案例：{len(failed)} 張（複製 {copied} 張圖至 {failed_dir}）")
    print(f"明細已寫入：{txt_path}")


def _print_summary(results: dict, total: int) -> None:
    s = _compute_summary(results)
    if not s:
        print("\n無有效結果可統計。")
        return

    print("\n" + "=" * 55)
    print("  GLM-OCR 日期辨識準確率評測結果")
    print("=" * 55)
    print(f"  評測張數   : {s['evaluated']} / {total}  （失敗：{s['errors']}）")
    print(f"  完整日期準確率 (全對): {s['full_date_acc']*100:.2f}%")
    print(f"  欄位準確率:")
    for cls, acc in s["field_acc"].items():
        bar = "█" * int(acc * 20)
        print(f"    {cls:<8}: {acc*100:6.2f}%  {bar}")
    print(f"  平均字元錯誤率 (CER) : {s['avg_cer']*100:.2f}%")
    print(f"  平均推論時間         : {s['avg_elapsed_sec']:.3f} s/張")
    print(f"\n  結果已存至：{RESULTS_PATH}")
    print("=" * 55)


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="GLM-OCR 日期辨識準確率評測")
    ap.add_argument("--limit",        type=int, default=None, help="只評測前 N 張（預設全部）")
    ap.add_argument("--resume",       action="store_true",    help="跳過已完成項目繼續執行")
    ap.add_argument("--export-failed", action="store_true",   help="從現有結果匯出失敗案例，不重跑推論")
    args = ap.parse_args()

    if args.export_failed:
        if not RESULTS_PATH.exists():
            print(f"找不到結果檔：{RESULTS_PATH}，請先執行評測。")
        else:
            with open(RESULTS_PATH, encoding="utf-8") as f:
                results = json.load(f).get("details", {})

            # 用目前最新的 field_hit 邏輯重新計算 hits / all_correct
            recomputed = 0
            for v in results.values():
                if "error" in v or "ocr" not in v:
                    continue
                tokens = extract_tokens(v["ocr"])
                new_hits = {cls: field_hit(gt, tokens) for cls, gt in v["gt"].items()}
                new_all  = all(new_hits.values())
                if new_hits != v.get("hits") or new_all != v.get("all_correct"):
                    v["hits"]        = new_hits
                    v["all_correct"] = new_all
                    recomputed += 1

            if recomputed:
                print(f"[重新計算] {recomputed} 筆結果已更新")
                _save(results)
            else:
                print("[重新計算] 所有結果與現有邏輯一致，無需更新")

            _print_summary(results, len(results))
            export_failed(results)
    else:
        run(args.limit, args.resume)
