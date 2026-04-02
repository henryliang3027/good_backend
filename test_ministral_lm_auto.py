from openai import OpenAI
import time
import subprocess
import urllib.request
import os
import signal

# ── 商品清單 ──────────────────────────────────────────────────────────────────
LABEL_NAMES = {
    0:  "冷山茶王",
    1:  "茶裏王台式綠茶",
    2:  "茶裏王日式無糖綠茶",
    3:  "茶裏王白毫烏龍",
    4:  "茶裏王半熟金萱",
    5:  "原萃台灣青茶",
    6:  "原萃烏龍茶",
    7:  "原萃鐵觀音",
    8:  "無加糖LP33機能優酪乳",
    9:  "御茶園特上檸檬茶",
    10: "每朝健康双纖綠茶",
    11: "每朝健康熟藏紅茶",
    12: "愛之味油切分解茶四季春風味",
    13: "濃韻無糖烏龍茶",
}

# ── 掃描清單（19 個，每個固定 6 品項）────────────────────────────────────────
SCAN_LISTS = [
    # 0 — 原始清單（取前 6 項）
    [("茶裏王台式綠茶", 2), ("原萃鐵觀音", 2), ("原萃烏龍茶", 1),
     ("茶裏王白毫烏龍", 2), ("原萃台灣青茶", 1), ("茶裏王日式無糖綠茶", 2)],

    # 1
    [("御茶園特上檸檬茶", 3), ("每朝健康双纖綠茶", 1), ("濃韻無糖烏龍茶", 4),
     ("茶裏王半熟金萱", 2), ("冷山茶王", 1), ("原萃鐵觀音", 5)],

    # 2
    [("茶裏王台式綠茶", 4), ("茶裏王日式無糖綠茶", 3), ("茶裏王白毫烏龍", 1),
     ("茶裏王半熟金萱", 5), ("無加糖LP33機能優酪乳", 2), ("冷山茶王", 3)],

    # 3 — 純原萃
    [("原萃台灣青茶", 6), ("原萃烏龍茶", 2), ("原萃鐵觀音", 4),
     ("濃韻無糖烏龍茶", 1), ("御茶園特上檸檬茶", 5), ("愛之味油切分解茶四季春風味", 3)],

    # 4 — 純每朝
    [("每朝健康双纖綠茶", 7), ("每朝健康熟藏紅茶", 3), ("無加糖LP33機能優酪乳", 4),
     ("冷山茶王", 2), ("茶裏王台式綠茶", 1), ("原萃烏龍茶", 6)],

    # 5
    [("愛之味油切分解茶四季春風味", 6), ("濃韻無糖烏龍茶", 1), ("御茶園特上檸檬茶", 2),
     ("每朝健康熟藏紅茶", 4), ("茶裏王白毫烏龍", 3), ("原萃台灣青茶", 5)],

    # 6
    [("冷山茶王", 4), ("每朝健康双纖綠茶", 3), ("每朝健康熟藏紅茶", 2),
     ("無加糖LP33機能優酪乳", 4), ("原萃烏龍茶", 1), ("茶裏王半熟金萱", 6)],

    # 7
    [("茶裏王台式綠茶", 1), ("原萃鐵觀音", 6), ("御茶園特上檸檬茶", 4),
     ("茶裏王半熟金萱", 2), ("愛之味油切分解茶四季春風味", 3), ("每朝健康熟藏紅茶", 5)],

    # 8
    [("濃韻無糖烏龍茶", 5), ("原萃台灣青茶", 2), ("每朝健康熟藏紅茶", 1),
     ("茶裏王日式無糖綠茶", 4), ("冷山茶王", 3), ("無加糖LP33機能優酪乳", 6)],

    # 9
    [("無加糖LP33機能優酪乳", 6), ("茶裏王白毫烏龍", 3), ("冷山茶王", 2),
     ("每朝健康双纖綠茶", 5), ("御茶園特上檸檬茶", 1), ("原萃鐵觀音", 4)],

    # 10
    [("原萃烏龍茶", 4), ("原萃台灣青茶", 4), ("原萃鐵觀音", 1),
     ("御茶園特上檸檬茶", 5), ("茶裏王日式無糖綠茶", 2), ("濃韻無糖烏龍茶", 3)],

    # 11
    [("茶裏王半熟金萱", 6), ("愛之味油切分解茶四季春風味", 1), ("濃韻無糖烏龍茶", 3),
     ("冷山茶王", 5), ("無加糖LP33機能優酪乳", 2), ("每朝健康双纖綠茶", 4)],

    # 12
    [("每朝健康熟藏紅茶", 4), ("茶裏王台式綠茶", 3), ("無加糖LP33機能優酪乳", 2),
     ("原萃烏龍茶", 5), ("御茶園特上檸檬茶", 1), ("茶裏王白毫烏龍", 6)],

    # 13 — 數量都是 1
    [("冷山茶王", 1), ("茶裏王台式綠茶", 1), ("原萃台灣青茶", 1),
     ("每朝健康双纖綠茶", 1), ("愛之味油切分解茶四季春風味", 1), ("濃韻無糖烏龍茶", 1)],

    # 14 — 數量都很高
    [("茶裏王日式無糖綠茶", 9), ("原萃鐵觀音", 8), ("濃韻無糖烏龍茶", 7),
     ("御茶園特上檸檬茶", 10), ("每朝健康熟藏紅茶", 6), ("無加糖LP33機能優酪乳", 5)],

    # 15
    [("茶裏王台式綠茶", 5), ("茶裏王日式無糖綠茶", 4), ("茶裏王白毫烏龍", 2),
     ("茶裏王半熟金萱", 3), ("冷山茶王", 1), ("原萃烏龍茶", 7)],

    # 16
    [("御茶園特上檸檬茶", 8), ("無加糖LP33機能優酪乳", 3), ("愛之味油切分解茶四季春風味", 2),
     ("每朝健康双纖綠茶", 6), ("原萃台灣青茶", 4), ("茶裏王半熟金萱", 1)],

    # 17
    [("濃韻無糖烏龍茶", 2), ("冷山茶王", 6), ("每朝健康熟藏紅茶", 5),
     ("茶裏王白毫烏龍", 4), ("原萃鐵觀音", 3), ("無加糖LP33機能優酪乳", 1)],

    # 18
    [("愛之味油切分解茶四季春風味", 4), ("每朝健康双纖綠茶", 2), ("茶裏王台式綠茶", 6),
     ("御茶園特上檸檬茶", 3), ("原萃烏龍茶", 5), ("冷山茶王", 4)],

    # 19
    [("茶裏王台式綠茶", 2), ("原萃鐵觀音", 1), ("茶裏王白毫烏龍", 2),
     ("原萃烏龍茶:", 2), ("茶裏王日式無糖綠茶", 2), ("原萃台灣青茶", 1)],
]

# ── 測試案例定義 ──────────────────────────────────────────────────────────────
# (scan_list_index, question, expected_keywords)
# expected_keywords: list of strings，每個都必須出現在回答中才算通過
TEST_CASES = [
    # --- 統計所有商品 ---
    (0,  "統計商品",          ["茶裏王台式綠茶 有 2 瓶", "原萃鐵觀音 有 2 瓶", "原萃烏龍茶 有 1 瓶",
                               "茶裏王白毫烏龍 有 2 瓶", "原萃台灣青茶 有 1 瓶", "茶裏王日式無糖綠茶 有 2 瓶"]),
    (2,  "統計商品",          ["茶裏王台式綠茶 有 4 瓶", "茶裏王日式無糖綠茶 有 3 瓶", "茶裏王白毫烏龍 有 1 瓶",
                               "茶裏王半熟金萱 有 5 瓶", "無加糖LP33機能優酪乳 有 2 瓶", "冷山茶王 有 3 瓶"]),
    (13, "統計商品",          ["冷山茶王 有 1 瓶", "茶裏王台式綠茶 有 1 瓶", "原萃台灣青茶 有 1 瓶",
                               "每朝健康双纖綠茶 有 1 瓶", "愛之味油切分解茶四季春風味 有 1 瓶", "濃韻無糖烏龍茶 有 1 瓶"]),

    # --- 品牌前綴查詢 (茶裏王) ---
    (0,  "有幾瓶茶裏王",      ["茶裏王台式綠茶 有 2 瓶", "茶裏王白毫烏龍 有 2 瓶", "茶裏王日式無糖綠茶 有 2 瓶"]),
    (2,  "有幾瓶茶裏王",      ["茶裏王台式綠茶 有 4 瓶", "茶裏王日式無糖綠茶 有 3 瓶", "茶裏王白毫烏龍 有 1 瓶", "茶裏王半熟金萱 有 5 瓶"]),
    (15, "有幾瓶茶裏王",      ["茶裏王台式綠茶 有 5 瓶", "茶裏王日式無糖綠茶 有 4 瓶", "茶裏王白毫烏龍 有 2 瓶", "茶裏王半熟金萱 有 3 瓶"]),

    # --- 品牌前綴查詢 (原萃) ---
    (0,  "有幾瓶原萃",        ["原萃鐵觀音 有 2 瓶", "原萃烏龍茶 有 1 瓶", "原萃台灣青茶 有 1 瓶"]),
    (3,  "有幾瓶原萃",        ["原萃台灣青茶 有 6 瓶", "原萃烏龍茶 有 2 瓶", "原萃鐵觀音 有 4 瓶"]),
    (10, "有幾瓶原萃",        ["原萃烏龍茶 有 4 瓶", "原萃台灣青茶 有 4 瓶", "原萃鐵觀音 有 1 瓶"]),

    # --- 品牌前綴查詢 (每朝) ---
    (4,  "有幾瓶每朝",        ["每朝健康双纖綠茶 有 7 瓶", "每朝健康熟藏紅茶 有 3 瓶"]),
    (6,  "有幾瓶每朝",        ["每朝健康双纖綠茶 有 3 瓶", "每朝健康熟藏紅茶 有 2 瓶"]),

    # --- 完整商品名稱查詢 ---
    (0,  "有幾瓶原萃鐵觀音",          ["原萃鐵觀音 有 2 瓶"]),
    (1,  "有幾瓶原萃鐵觀音",          ["原萃鐵觀音 有 5 瓶"]),
    (9,  "有幾瓶原萃鐵觀音",          ["原萃鐵觀音 有 4 瓶"]),
    (14, "有幾瓶茶裏王日式無糖綠茶",  ["茶裏王日式無糖綠茶 有 9 瓶"]),
    (14, "有幾瓶濃韻無糖烏龍茶",      ["濃韻無糖烏龍茶 有 7 瓶"]),
    (5,  "有幾瓶愛之味油切分解茶四季春風味", ["愛之味油切分解茶四季春風味 有 6 瓶"]),
    (16, "有幾瓶御茶園特上檸檬茶",    ["御茶園特上檸檬茶 有 8 瓶"]),
    (8,  "有幾瓶無加糖LP33機能優酪乳",["無加糖LP33機能優酪乳 有 6 瓶"]),

    # --- 查無商品 ---
    (0,  "有幾瓶冷山茶王",            ["沒有找到您指定的商品"]),
    (0,  "有幾瓶御茶園特上檸檬茶",    ["沒有找到您指定的商品"]),
    (13, "有幾瓶原萃",                ["原萃台灣青茶 有 1 瓶"]),
    (4,  "有幾瓶冷山茶王",            ["冷山茶王 有 2 瓶"]),
]

# ── System prompt builder ─────────────────────────────────────────────────────
SYSTEM_PROMPT_RULES = """
【輸出格式——絕對遵守】
每筆商品資訊必須嚴格使用下列格式，注意「有」字與空格，禁止使用冒號（: 或 ：）：
  [商品名稱] 有 [數量] 瓶

正確：茶裏王台式綠茶 有 2 瓶
禁止：茶裏王台式綠茶: 2 瓶
禁止：茶裏王台式綠茶：2 瓶
禁止：茶裏王台式綠茶 2 瓶

【回答規則】
1. 「統計商品」——列出清單中所有商品，每行一個，格式如上。不加任何標題或列點符號。

2. 「有幾瓶 [品牌]」——品牌前綴查詢（如：茶裏王、原萃、每朝）：
   - 從掃描清單中找出所有名稱「以該品牌為開頭」的商品。
   - 每行一個，格式如上。必須列出所有符合的商品，不可遺漏任何一項。
   - 禁止列出名稱不以該品牌為開頭的商品。例如詢問「原萃」時，不可列出「茶裏王」開頭的商品。
   - 若清單中完全沒有符合的商品，僅回答：沒有找到您指定的商品

3. 「有幾瓶 [完整商品名稱]」——完整名稱查詢（問題中包含完整商品名，例如「有幾瓶原萃鐵觀音」）：
   - 只回答該指定商品，不可列出其他商品。
   - 若清單中有該商品，回答：[商品名稱] 有 [數量] 瓶
   - 若清單中沒有該商品，回答：沒有找到您指定的商品

4. 禁止輸出任何額外說明、前言或結尾客套話。
5. 必須使用繁體中文。
6. 若遇到語音辨識諧音詞，自動對應到清單中最相似的商品名稱。
7. 商品名稱必須與掃描清單完全一致，逐字照抄，禁止增加、刪除或重複任何文字。
"""


def build_system_prompt(scan_list: list[tuple[str, int]]) -> str:
    items = "\n".join(f"- {name}: {qty} 瓶" for name, qty in scan_list)
    return f"你是一位專業的超商貨架分析員。請根據以下掃描結果清單回答用戶問題。\n\n【掃描結果清單】\n{items}\n{SYSTEM_PROMPT_RULES}"


# ── llama-server ──────────────────────────────────────────────────────────────
LLAMA_SERVER_CMD = [
    "./llama.cpp/build/bin/llama-server",
    "-m", "ministral/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf",
    "--mmproj", "ministral/mmproj-F16.gguf",
    "--host", "0.0.0.0",
    "--port", "8881",
    "--ctx-size", "4096",
    "-ngl", "-1",
]

client = OpenAI(
    base_url="http://127.0.0.1:8881/v1",
    api_key="no-key-needed",
)


def start_server() -> subprocess.Popen:
    proc = subprocess.Popen(
        LLAMA_SERVER_CMD,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    print("等待 llama-server 啟動...")
    for _ in range(60):
        try:
            urllib.request.urlopen("http://127.0.0.1:8881/health", timeout=2)
            print("Server 已就緒\n")
            return proc
        except Exception:
            time.sleep(2)
    raise RuntimeError("llama-server 啟動逾時")


def ask(scan_list: list[tuple[str, int]], question: str) -> tuple[str, float]:
    system_prompt = build_system_prompt(scan_list)
    t0 = time.time()
    response = client.chat.completions.create(
        model="ministral_3_3b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [{"type": "text", "text": question}]},
        ],
        temperature=0,
    )
    elapsed = round(time.time() - t0, 3)
    return response.choices[0].message.content.strip(), elapsed


def run_tests():
    passed = 0
    failed = 0
    results = []

    print("=" * 60)
    print(f"執行 {len(TEST_CASES)} 個測試案例")
    print("=" * 60)

    for i, (list_idx, question, expected_keywords) in enumerate(TEST_CASES):
        scan_list = SCAN_LISTS[list_idx]
        answer, elapsed = ask(scan_list, question)

        

        answer_lines = [line for line in answer.splitlines() if line]

        print(answer)
        print(answer_lines)
        missing = [kw for kw in expected_keywords if kw not in answer]
        extra   = [line for line in answer_lines if line not in expected_keywords]
        ok = not missing and not extra
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1

        results.append((i + 1, list_idx, question, answer, elapsed, ok, expected_keywords))

        # 即時輸出
        print(f"\n[{i+1:02d}/{len(TEST_CASES)}] {status}  ({elapsed}s)")
        print(f"  清單 #{list_idx}: {[f'{n}×{q}' for n, q in scan_list]}")
        print(f"  問題: {question}")
        if not ok:
            print(f"  回答: {answer}")
            print(f"  缺少關鍵字: {missing}")
            print(f"  多餘內容: {extra}")

    # 總結
    print("\n" + "=" * 60)
    print(f"結果: {passed} PASS / {failed} FAIL  (共 {len(TEST_CASES)} 案例)")
    print("=" * 60)

    if failed > 0:
        print("\n失敗案例明細:")
        for no, list_idx, question, answer, elapsed, ok, expected_keywords in results:
            if not ok:
                answer_lines = [line for line in answer.splitlines() if line]
                missing = [kw for kw in expected_keywords if kw not in answer]
                extra   = [line for line in answer_lines if line not in expected_keywords]
                print(f"  #{no:02d} 清單#{list_idx} | Q: {question}")
                print(f"       缺少: {missing}")
                print(f"       多餘: {extra}")
                print(f"       回答: {answer!r}")

    return failed == 0


if __name__ == "__main__":
    proc = start_server()
    try:
        success = run_tests()
    finally:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        print("\nllama-server 已關閉")

    exit(0 if success else 1)
