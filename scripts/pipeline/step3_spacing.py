"""
Step 3 - 줄 내 콩 파종 간격 산출.

Step 2가 각 두둑에서 파종 줄 2개의 perp 위치를 잡았으니, 여기선
각 줄 안에서 SAM 잎을 ridge 방향으로 사영해 인접 잎 간격 계산.

흐름:
  1. rows.npz + ridges.npz 로드
  2. 각 줄마다:
      - 밴드 필터: |perp - row_perp| ≤ ROW_HALF_CM
      - ridge 축 사영 → 1D 위치
      - 인접 간격 계산 → median = 파종 간격
      - 결주 판정: gap > 1.5 × median 이면 결주 구간
  3. 통계 저장 + 시각화 (히스토그램, 유형별 비교, 예시 도트플롯)

사용:
  python -u scripts/pipeline/step3_spacing.py GJSM-1-1_Smart

출력:
  result/pipeline/{field}/spacing.npz
  result/pipeline/{field}/step3_spacing.png
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.common import (
    load_field_npz, leaf_centroids, get_field_dir, setup_korean_font,
)


# ---- 파라미터 ----
ROW_HALF_CM = 8            # 파종 줄 밴드 반경 (총 16cm)
MIN_LEAVES_FOR_STATS = 5   # 통계 낼 수 있는 최소 잎 개수
SPACING_SPEC_CM = 20       # 스펙
SPACING_TOL_CM = 5         # 허용 ±5cm → 15~25cm 적합
MISS_MULT = 1.5            # median × 1.5 넘으면 결주 gap
DEDUP_MIN_GAP_CM = 8       # 8cm 이내 인접 잎은 같은 콩 개체로 병합
                            # (SAM이 큰 잎을 여러 마스크로 잡는 문제 보정)


def load_ridges(field_dir: Path):
    npz = np.load(field_dir / "ridges.npz", allow_pickle=True)
    return dict(
        origin=npz["origin"], perp=npz["perp"], ridge_dir=npz["ridge_dir"],
        px_to_cm=float(npz["px_to_cm"][0]),
        ridge_arr=npz["ridge_arr"], ridge_types=npz["ridge_types"],
    )


def load_rows(field_dir: Path):
    npz = np.load(field_dir / "rows.npz", allow_pickle=True)
    return dict(row_arr=npz["row_arr"], type_arr=npz["type_arr"])


def analyze_row(sam_perp, sam_ridge, row_perp_px, ridge_min_px, ridge_max_px,
                 px_to_cm: float) -> dict:
    """
    한 파종 줄 통계.
      sam_perp/sam_ridge: 필지 전체 SAM 잎의 perp/ridge 좌표 (px)
      row_perp_px: 이 줄의 perp 좌표 (px)
      ridge_min/max_px: 이 두둑의 ridge 축 범위
    """
    half_px = ROW_HALF_CM / px_to_cm
    mask = (np.abs(sam_perp - row_perp_px) <= half_px) \
           & (sam_ridge >= ridge_min_px) & (sam_ridge <= ridge_max_px)
    n = int(mask.sum())
    if n < 2:
        return dict(n=n, positions_cm=np.array([]), gaps_cm=np.array([]),
                     median_cm=np.nan, mean_cm=np.nan, std_cm=np.nan,
                     miss_gaps=0, expected_plants=0, emergence_pct=np.nan,
                     length_cm=(ridge_max_px - ridge_min_px) * px_to_cm)

    positions_px = np.sort(sam_ridge[mask])
    positions_cm_raw = positions_px * px_to_cm

    # 개체 dedup: 인접 잎이 DEDUP_MIN_GAP_CM 이내면 같은 개체 → 뒤 잎 제거
    if len(positions_cm_raw) >= 2:
        kept = [positions_cm_raw[0]]
        for p in positions_cm_raw[1:]:
            if p - kept[-1] >= DEDUP_MIN_GAP_CM:
                kept.append(p)
        positions_cm = np.array(kept)
    else:
        positions_cm = positions_cm_raw
    n = len(positions_cm)
    if n < 2:
        return dict(n=n, positions_cm=positions_cm, gaps_cm=np.array([]),
                     median_cm=np.nan, mean_cm=np.nan, std_cm=np.nan,
                     miss_gaps=0, expected_plants=0, emergence_pct=np.nan,
                     length_cm=(ridge_max_px - ridge_min_px) * px_to_cm)

    gaps_cm = np.diff(positions_cm)
    med = float(np.median(gaps_cm))
    mean_ = float(gaps_cm.mean())
    std_ = float(gaps_cm.std())

    # 결주 (결주 gap = median * MISS_MULT 초과)
    miss_thresh = med * MISS_MULT
    miss_mask = gaps_cm > miss_thresh
    n_miss_gaps = int(miss_mask.sum())

    # 입모율 = 실측 개체 / 기대 개체(길이/스펙 20cm)
    length_cm = (ridge_max_px - ridge_min_px) * px_to_cm
    expected = max(int(round(length_cm / SPACING_SPEC_CM)), 1)
    emergence_pct = 100.0 * n / expected

    return dict(
        n=n, positions_cm=positions_cm, gaps_cm=gaps_cm,
        median_cm=med, mean_cm=mean_, std_cm=std_,
        miss_gaps=n_miss_gaps, expected_plants=expected,
        emergence_pct=emergence_pct, length_cm=length_cm,
    )


def visualize(res: dict, out_png: Path):
    setup_korean_font()
    fig = plt.figure(figsize=(24, 15))
    gs = fig.add_gridspec(3, 4)

    rgb = res["rgb_disp"]; rgb_step = res["rgb_step"]
    origin = res["origin"]; perp = res["perp"]; ridge_dir = res["ridge_dir"]
    px_to_cm = res["px_to_cm"]

    all_rows = res["row_stats"]        # list of dict with type + stats
    narrow_rows = [r for r in all_rows if r["type_final"] == "narrow"]
    wide_rows = [r for r in all_rows if r["type_final"] == "wide"]
    other_rows = [r for r in all_rows if r["type_final"] == "other"]

    def gather(rows, key):
        return [r[key] for r in rows if not np.isnan(r.get(key, np.nan))]

    # (1) 필지 전체 RGB + 잎 색상 (row별)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.imshow(rgb)
    # 밴드 안 잎 색: narrow=주황, wide=보라
    sam_pts = res["sam_pts"]
    ax1.scatter(sam_pts[:, 1] / rgb_step, sam_pts[:, 0] / rgb_step,
                 s=0.7, c="lime", alpha=0.35, edgecolor="none")
    ax1.set_title("필지 + SAM 잎 (초록)", fontsize=12, fontweight="bold")
    ax1.axis("off")

    # (2) 파종 간격 분포 히스토그램
    ax2 = fig.add_subplot(gs[0, 2])
    med_narrow = gather(narrow_rows, "median_cm")
    med_wide = gather(wide_rows, "median_cm")
    bins = np.arange(0, 60, 2)
    if med_narrow:
        ax2.hist(med_narrow, bins=bins, color="orange", alpha=0.7,
                 edgecolor="darkorange", label=f"narrow (n={len(med_narrow)})")
    if med_wide:
        ax2.hist(med_wide, bins=bins, color="purple", alpha=0.5,
                 edgecolor="indigo", label=f"wide (n={len(med_wide)})")
    ax2.axvspan(SPACING_SPEC_CM - SPACING_TOL_CM,
                 SPACING_SPEC_CM + SPACING_TOL_CM,
                 color="green", alpha=0.15,
                 label=f"적합 {SPACING_SPEC_CM}±{SPACING_TOL_CM}cm")
    ax2.axvline(SPACING_SPEC_CM, color="green", lw=1.5, ls="--")
    ax2.set_xlabel("줄별 median 파종 간격 (cm)"); ax2.set_ylabel("줄 수")
    ax2.set_title(f"줄별 median 간격 분포", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

    # (3) 요약 텍스트
    ax3 = fig.add_subplot(gs[0, 3])
    ax3.axis("off")
    def stats_line(rows, label):
        med_all = gather(rows, "median_cm")
        emg = gather(rows, "emergence_pct")
        return (f"- {label} (n={len(rows)}): "
                f"median 간격 {np.median(med_all):.1f}cm, "
                f"입모율 median {np.median(emg):.1f}%"
                if med_all else f"- {label}: 데이터 없음")
    total_leaves = sum(r["n"] for r in all_rows)
    lines = [
        f"필지: {res['field']}",
        f"두둑 {res['n_ridges']}, 파종 줄 {len(all_rows)}",
        f"밴드 안 잎 총 {total_leaves:,}",
        "",
        stats_line(narrow_rows, "narrow (좁은)"),
        stats_line(wide_rows,  "wide   (넓은)"),
        stats_line(other_rows, "other  (벗어남)"),
        "",
        f"스펙: 파종 간격 {SPACING_SPEC_CM}cm "
        f"(적합 ±{SPACING_TOL_CM}cm)",
    ]
    # 적합 줄 개수
    ok_narrow = sum(SPACING_SPEC_CM - SPACING_TOL_CM
                     <= r["median_cm"] <= SPACING_SPEC_CM + SPACING_TOL_CM
                     for r in narrow_rows if not np.isnan(r["median_cm"]))
    ok_wide = sum(SPACING_SPEC_CM - SPACING_TOL_CM
                   <= r["median_cm"] <= SPACING_SPEC_CM + SPACING_TOL_CM
                   for r in wide_rows if not np.isnan(r["median_cm"]))
    lines += [
        f"적합 줄 (narrow): {ok_narrow}/{len(narrow_rows)}",
        f"적합 줄 (wide)  : {ok_wide}/{len(wide_rows)}",
    ]
    ax3.text(0.02, 0.98, "\n".join(lines), fontsize=12,
             va="top", ha="left", transform=ax3.transAxes)

    # (4) 예시 줄 도트플롯 (narrow 1, wide 1, 결주 많은 1)
    example_rows = []
    for r in narrow_rows:
        if r["n"] >= MIN_LEAVES_FOR_STATS:
            example_rows.append(r); break
    for r in wide_rows:
        if r["n"] >= MIN_LEAVES_FOR_STATS:
            example_rows.append(r); break
    # 결주 많은 줄
    with_miss = sorted([r for r in all_rows if r["miss_gaps"] > 0],
                        key=lambda x: -x["miss_gaps"])
    if with_miss:
        example_rows.append(with_miss[0])
    example_rows = example_rows[:3]

    for i, r in enumerate(example_rows):
        ax = fig.add_subplot(gs[1 + i // 2, (i % 2) * 2: (i % 2) * 2 + 2])
        pos_cm = r["positions_cm"] - r["positions_cm"].min()
        ax.scatter(pos_cm, np.zeros_like(pos_cm), s=25,
                    c="lime", edgecolor="darkgreen", zorder=3)
        # 결주 gap 강조
        if len(r["gaps_cm"]):
            gaps = r["gaps_cm"]
            for j, g in enumerate(gaps):
                x1 = pos_cm[j]; x2 = pos_cm[j + 1]
                if g > r["median_cm"] * MISS_MULT:
                    ax.axvspan(x1, x2, ymin=0.4, ymax=0.6,
                                color="red", alpha=0.35)
                    ax.text((x1 + x2) / 2, 0.5, f"{g:.0f}", fontsize=8,
                             color="red", ha="center", va="center",
                             fontweight="bold")
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_ylim(-1, 1); ax.set_yticks([])
        ax.set_xlabel("줄 방향 위치 (cm)")
        ax.set_title(
            f"두둑 {r['ridge_id']} row {r['row_idx']} ({r['type_final']}) - "
            f"잎 {r['n']}, median {r['median_cm']:.1f}cm, "
            f"입모율 {r['emergence_pct']:.0f}%, 결주 {r['miss_gaps']}",
            fontsize=11)
        ax.grid(alpha=0.3, axis="x")

    plt.suptitle(f"Step 3: 줄 내 콩 파종 간격 - {res['field']}",
                 fontsize=15, fontweight="bold", y=1.005)
    plt.tight_layout()
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close()


def main():
    if len(sys.argv) < 2:
        print("사용: python step3_spacing.py <field_name>")
        sys.exit(1)
    field = sys.argv[1]
    t0 = time.time()

    print(f"[{field}] npz 로드")
    d = load_field_npz(field)
    sam_pts = leaf_centroids(d["leaves"])

    field_dir = get_field_dir(field)
    R = load_ridges(field_dir)
    S = load_rows(field_dir)
    origin = R["origin"]; perp = R["perp"]; ridge_dir = R["ridge_dir"]
    px_to_cm = R["px_to_cm"]
    ridge_arr = R["ridge_arr"]; ridge_types = R["ridge_types"]
    row_arr = S["row_arr"]; type_arr = S["type_arr"]
    print(f"   두둑 {len(ridge_arr)}, 파종 줄 후보 {len(row_arr)}")

    # SAM 잎 좌표 사영
    sam_rel = sam_pts - origin
    sam_perp = sam_rel @ perp
    sam_ridge = sam_rel @ ridge_dir

    print(f"[{field}] 줄별 사영·간격 계산")
    # ridge_id → ridge info
    ridge_lut = {int(row[0]): dict(
        ridge_min_px=float(row[6]), ridge_max_px=float(row[7]),
        length_cm=float(row[8])) for row in ridge_arr}
    # row_arr: [ridge_id, n_peaks, p1_px, p2_px, gap_cm, n_leaves]
    all_rows = []
    for i, row in enumerate(row_arr):
        rid = int(row[0])
        n_peaks = int(row[1])
        ridge_info = ridge_lut[rid]
        rtype = str(type_arr[i])
        if n_peaks == 0:
            continue
        peaks_px = [row[2], row[3]][:n_peaks]
        for j, p_perp_px in enumerate(peaks_px):
            if np.isnan(p_perp_px):
                continue
            stats = analyze_row(sam_perp, sam_ridge, float(p_perp_px),
                                 ridge_info["ridge_min_px"],
                                 ridge_info["ridge_max_px"], px_to_cm)
            stats["ridge_id"] = rid
            stats["row_idx"] = j
            stats["type_final"] = rtype
            stats["row_perp_px"] = float(p_perp_px)
            all_rows.append(stats)

    n_valid = sum(1 for r in all_rows if r["n"] >= MIN_LEAVES_FOR_STATS)
    print(f"   줄 총 {len(all_rows)} (유효 통계 {n_valid})")

    # 유형별 요약
    def gather(rows, key):
        return [r[key] for r in rows if not np.isnan(r.get(key, np.nan))]
    for tlabel in ["narrow", "wide", "other"]:
        rs = [r for r in all_rows if r["type_final"] == tlabel]
        med_all = gather(rs, "median_cm")
        emg = gather(rs, "emergence_pct")
        if med_all:
            print(f"   {tlabel:6s}: {len(rs):3d} 줄, "
                  f"median 간격 {np.median(med_all):5.1f}cm, "
                  f"입모율 median {np.median(emg):5.1f}%")

    # 저장
    out_npz = field_dir / "spacing.npz"
    # rows_stats_arr: [ridge_id, row_idx, n, median_cm, mean_cm, std_cm,
    #                   miss_gaps, expected_plants, emergence_pct, length_cm,
    #                   row_perp_px]
    stats_arr = np.array([[
        r["ridge_id"], r["row_idx"], r["n"], r["median_cm"], r["mean_cm"],
        r["std_cm"], r["miss_gaps"], r["expected_plants"],
        r["emergence_pct"], r["length_cm"], r["row_perp_px"],
    ] for r in all_rows], dtype=np.float32)
    stats_types = np.array([r["type_final"] for r in all_rows])
    # 각 줄별 잎 위치 (variable length → object array)
    positions_obj = np.array([r["positions_cm"] for r in all_rows], dtype=object)
    np.savez_compressed(out_npz,
                         stats_arr=stats_arr,
                         stats_types=stats_types,
                         positions_obj=positions_obj)
    print(f"[{field}] 저장: {out_npz.name}")

    out_png = field_dir / "step3_spacing.png"
    visualize({
        "field": field, "rgb_disp": d["rgb_disp"], "rgb_step": d["rgb_step"],
        "origin": origin, "perp": perp, "ridge_dir": ridge_dir,
        "px_to_cm": px_to_cm,
        "n_ridges": len(ridge_arr), "sam_pts": sam_pts,
        "row_stats": all_rows,
    }, out_png)
    print(f"[{field}] 저장: {out_png.name}")
    print(f"[{field}] 완료 ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
