"""
콩 입모율 분석 전체 파이프라인 - 4개 스크립트 순차 실행.

실행:
    cd C:/Users/user/Desktop/분석프로젝트/Soybean_Emergence_Rate
    python -u scripts/run_all.py

산출물은 result/fields/ 와 result/emergence/ 에 생성.
소요 시간: 약 80분 (대부분 02_analyze_emergence.py).
"""
from __future__ import annotations
from pathlib import Path
import os
import subprocess
import sys
import time

# Windows cp949 인코딩 문제 회피
os.environ["PYTHONIOENCODING"] = "utf-8"
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = [
    ("01_crop_fields.py",         "필지별 TIF 크롭",       "약 8분"),
    ("02_analyze_emergence.py",   "입모율 분석",           "약 70분"),
    ("03_postprocess_visualize.py","PPT 시각화 + 파종기 비교","약 1분"),
    ("04_detail_overlay.py",      "RGB 상세 오버레이 + CSV 보강","약 2분"),
]


def main():
    print("="*70)
    print("콩 입모율 분석 전체 파이프라인")
    print(f"작업 폴더: {ROOT}")
    print(f"파이썬: {sys.executable}")
    print("="*70)

    t_total = time.time()
    for fname, desc, est in SCRIPTS:
        path = ROOT / "scripts" / fname
        if not path.exists():
            print(f"\n[SKIP] {fname} (파일 없음)")
            continue
        print(f"\n{'='*70}")
        print(f"  [실행] {fname}  - {desc}  (예상 {est})")
        print(f"{'='*70}")
        t0 = time.time()
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run([sys.executable, "-u", str(path)],
                               cwd=str(ROOT), env=env)
        if result.returncode != 0:
            print(f"\n[!] {fname} 실패 (exit {result.returncode})")
            sys.exit(result.returncode)
        print(f"\n[완료] {fname}  ({time.time()-t0:.1f}s)")

    print(f"\n{'='*70}")
    print(f"전체 완료: {(time.time()-t_total)/60:.1f}분")
    print(f"결과: {ROOT/'result'}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
