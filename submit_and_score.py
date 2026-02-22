#!/usr/bin/env python3
"""
Upload submission zip to bounty.autohdr.com, poll for score, and print results.
"""
import csv
import json
import time
from pathlib import Path

import requests

BASE_URL = "https://bounty.autohdr.com"
TEAM_NAME = "DarshanSolo"

KAGGLE_USERNAME = "darshanarao"
EMAIL = "rao.darshan@gmail.com"
GITHUB_REPO = "darshanrao/auto_hdr_rewamp"

_SCRIPT_DIR = Path(__file__).resolve().parent
ZIP_PATH = _SCRIPT_DIR / "submission_bicubic_tta_512.zip"
WORKING_DIR = _SCRIPT_DIR / "working"

POLL_EVERY_SECONDS = 3
TIMEOUT_SECONDS = 600 * 2  # 20 minutes


def join_url(base, path):
    return base.rstrip("/") + path


def safe_json(resp):
    try:
        return resp.json()
    except Exception:
        return {"_raw": resp.text}


def main():
    WORKING_DIR.mkdir(parents=True, exist_ok=True)

    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"Zip not found: {ZIP_PATH}")

    session = requests.Session()

    print("1) Getting upload URL...")
    resp = session.post(
        join_url(BASE_URL, "/api/upload-url"),
        headers={"Content-Type": "application/json"},
        data=json.dumps({"team_name": TEAM_NAME.strip()}),
        timeout=10,
    )
    data = safe_json(resp)
    if not resp.ok:
        raise RuntimeError(f"upload-url failed ({resp.status_code}): {data}")

    upload_url = data.get("upload_url")
    s3_key = data.get("s3_key")
    if not upload_url or not s3_key:
        raise RuntimeError(f"Missing upload_url/s3_key in response: {data}")

    print("2) Uploading zip to S3...")
    zip_bytes = ZIP_PATH.read_bytes()
    put = requests.put(
        upload_url,
        headers={"Content-Type": "application/zip"},
        data=zip_bytes,
        timeout=420,
    )
    if not put.ok:
        raise RuntimeError(f"PUT upload failed ({put.status_code}): {put.text[:500]}")

    print("3) Submitting for scoring...")
    payload = {
        "s3_key": s3_key,
        "team_name": TEAM_NAME.strip(),
        "kaggle_username": KAGGLE_USERNAME.strip(),
        "email": EMAIL.strip(),
        "github_repo": GITHUB_REPO.strip(),
    }
    resp = session.post(
        join_url(BASE_URL, "/api/score"),
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=60,
    )
    data = safe_json(resp)
    if not resp.ok:
        raise RuntimeError(f"submit failed ({resp.status_code}): {data}")

    request_id = data.get("request_id")
    if not request_id:
        raise RuntimeError(f"Missing request_id in response: {data}")

    print("4) Polling status...")
    elapsed = 0
    while True:
        time.sleep(POLL_EVERY_SECONDS)
        elapsed += POLL_EVERY_SECONDS

        resp = session.get(
            join_url(BASE_URL, f"/api/score?request_id={request_id}"),
            timeout=60,
        )
        data = safe_json(resp)
        if not resp.ok:
            raise RuntimeError(f"poll failed ({resp.status_code}): {data}")

        out_json = WORKING_DIR / "submission.json"
        out_json.write_text(json.dumps(data, indent=2), encoding="utf-8")

        status = data.get("status")
        if status == "COMPLETED":
            if not data.get("success", False):
                raise RuntimeError(
                    data.get("detail") or "Scoring failed (COMPLETED but success=false)"
                )

            print("\nCOMPLETED")

            csv_text = data.get("csv", "")
            if csv_text:
                out_csv = WORKING_DIR / "submission.csv"
                out_csv.write_text(csv_text, encoding="utf-8")
                print(f"Saved CSV to: {out_csv}")
            else:
                print("No csv returned.")
            break

        if status == "FAILED" or data.get("detail"):
            raise RuntimeError(data.get("detail") or "Scoring failed")

        if elapsed > TIMEOUT_SECONDS:
            raise TimeoutError("Scoring timed out")

        print(f"  {status} ({elapsed}s)")

    # Parse scores from CSV
    csv_path = WORKING_DIR / "submission.csv"
    if csv_path.exists():
        scores = []
        with csv_path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                s = row.get("score")
                if s is None or s.strip() == "":
                    continue
                try:
                    scores.append(float(s))
                except ValueError:
                    continue

        if scores:
            print(f"\n--- Score ---")
            print(f"Average score: {sum(scores) / len(scores):.4f}")
        else:
            print("\nNo scores found in CSV")
    else:
        print("\nCSV not saved, skipping score average")

    # Average params from json summary
    params = [
        "edge_similarity",
        "line_straightness",
        "gradient_orientation",
        "ssim",
        "l1_pixel_diff",
        "max_regional_diff",
        "gt_edge_density",
    ]
    summary = data.get("summary")
    if summary:
        params_array = {p: [] for p in params}
        images = summary.get("per_image", [])
        for img in images:
            metrics = img.get("metrics", {})
            for p in params:
                if p in metrics:
                    params_array[p].append(metrics[p])
        averages = {
            p: sum(params_array[p]) / len(params_array[p]) if params_array[p] else 0
            for p in params_array
        }
        print("\n--- Average params ---")
        for p in params:
            if p in averages:
                print(f"  {p}: {averages[p]:.4f}")


if __name__ == "__main__":
    main()
