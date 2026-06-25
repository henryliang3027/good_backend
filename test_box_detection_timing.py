"""
Compare timing between /box_date_detection (IoU matching) and
/box_date_detection2 (per-box crop detection).
"""

import base64
import time

import requests


def load_image_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def call_endpoint(url: str, image_b64: str) -> tuple[float, dict]:
    payload = {"image_base64": image_b64}
    t0 = time.perf_counter()
    resp = requests.post(url, json=payload, timeout=120)
    elapsed = time.perf_counter() - t0
    resp.raise_for_status()
    return elapsed, resp.json()


def run_benchmark(image_path: str, runs: int, base_url: str):
    print(f"Loading image: {image_path}")
    image_b64 = load_image_b64(image_path)
    print(f"Image size: {len(image_b64) / 1024:.1f} KB (base64)\n")

    url1 = f"{base_url}/box_date_detection2"

    times1: list[float] = []

    for run in range(1, runs + 1):
        print(f"--- Run {run}/{runs} ---")

        elapsed1, result1 = call_endpoint(url1, image_b64)
        times1.append(elapsed1)
        boxes1 = result1.get("data", [])
        print(f"  /box_date_detection  : {elapsed1:.3f}s  |  boxes={len(boxes1)}")


        if run == 1:
            print(f"\n  [result sample]")
            for item in boxes1:
                print(f"    {item['name']} → date={item['date']}  date_bbox={item['date_bbox']}")

    print("\n========== Summary ==========")
    avg1 = sum(times1) / len(times1)
    min1, max1 = min(times1), max(times1)

    print(f"Runs: {runs}")
    print()
    print(f"  /box_date_detection  (IoU matching on full image)")
    print(f"    avg={avg1:.3f}s  min={min1:.3f}s  max={max1:.3f}s")



if __name__ == "__main__":
    IMAGE_PATH = "original_box_images/input_20260605_090310_714022.jpg"
    RUNS = 1
    BASE_URL = "http://127.0.0.1:8888"

    run_benchmark(IMAGE_PATH, RUNS, BASE_URL)
