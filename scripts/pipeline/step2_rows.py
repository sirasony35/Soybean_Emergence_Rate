"""
Step 2 - 두둑별 dual-row (2줄) 검출 + 조간 산출.

Step 1이 두둑 중심선을 잡았으므로, 여기선 각 두둑 밴드 안에서만 sub-scan.

핵심 신호:
  각 두둑 위에는 파종기가 만든 어두운 홈 (dark stripe) 2줄이 있음.
  → gray 반전(dark = 큼) + SAM 잎 밀도 결합 프로파일 → find_peaks(최대 2)

흐름:
  1. ridges.npz + npz 로드
  2. 각 두둑 (id r):
      a. 두둑 중심 기준 ±SEARCH_HALF_CM 밴드 안의 필지 픽셀 뽑기
      b. 두둑 방향으로 perp 축 sub-profile (fine bin 0.2cm)
      c. Gray 반전 + SAM 밀도 결합 후 bandpass (5~50cm)
      d. find_peaks (min_dist=15cm, top-2 by height)
      e. 2 peak 발견 시 조간 = 두 peak 거리
  3. 두둑 유형 재분류:
      narrow: 조간 20~45cm
      wide:   조간 55~85cm
      other:  조간 벗어남 or 2 peak 미검출
  4. 저장 rows.npz + 시각화

사용:
  python -u scripts/pipeline/step2_rows.py GJSM-1-1_Smart

출력:
  result/pipeline/{field}/rows.npz
  result/pipeline/{field}/step2_rows.png
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

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
SEARCH_HALF_CM = 50            # 두둑 중심 ±50cm (총 100cm) 안에서 dual-row 탐색
SUB_BIN_CM = 0.3
SUB_SMOOTH_CM = 1.5
BANDPASS_LO_CM = 5             # 5cm 이하 노이즈 제거
BANDPASS_HI_CM = 90            # 90cm 이상 baseline 제거
ROW_MIN_SPACING_CM = 15        # 두 파종 줄 최소 간격
GRAY_INVERT_WEIGHT = 1.0       # gray 반전 신호 가중 (dark = row)
SAM_WEIGHT = 0.8               # SAM 밀도 가중

# 조간 유형 분류
NARROW_MIN_CM, NARROW_MAX_CM = 20, 45     # 스펙 30cm ±15
WIDE_MIN_CM, WIDE_MAX_CM = 55, 85         # 스펙 70cm ±15


def rgb_to_gray_and_mask(rgb: np.ndarray):
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    mask = (r + g + b) > 5
    return gray, mask


def bandpass_1d(profile, bin_cm, lo_cm, hi_cm):
    sigma_low = lo_cm / 2 / bin_cm
    sigma_high = hi_cm / 2 / bin_cm
    smooth = gaussian_filter1d(profile, sigma=sigma_low)
    baseline = gaussian_filter1d(profile, sigma=sigma_high)
    return smooth - baseline


def norm01(x):
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    lo, hi = np.percentile(x, [2, 98])
    if hi - lo < 1e-6:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0, 1)


def compute_ridge_sub_profile(
    ridge: dict, origin, perp, ridge_dir, px_to_cm: float,
    gray_coords_c, gray_vals, gray_mask,
    sam_perp_all, sam_ridge_all,
) -> dict:
    """
    한 두둑의 밴드 안에서 perp 축 fine profile 계산.
    return dict {perp_cm, gray_dark_norm, sam_norm, combined, peaks_perp_px, gap_cm, ...}
    """
    center = ridge["center_perp_px"]
    ridge_min = ridge["ridge_min_px"]
    ridge_max = ridge["ridge_max_px"]
    search_half_px = SEARCH_HALF_CM / px_to_cm

    # 필지 픽셀 mask: perp ±search_half AND ridge 축은 두둑 길이 범위 안
    gp_perp = gray_coords_c @ perp
    gp_ridge = gray_coords_c @ ridge_dir
    in_band = gray_mask & (np.abs(gp_perp - center) <= search_half_px) \
              & (gp_ridge >= ridge_min) & (gp_ridge <= ridge_max)

    if in_band.sum() < 100:
        return dict(perp_cm=np.array([]), status="empty")

    # 상대 perp (두둑 중심 기준)
    rel_perp = gp_perp[in_band] - center
    gray_v = gray_vals[in_band]

    bin_px = SUB_BIN_CM / px_to_cm
    lo = -search_half_px; hi = search_half_px
    edges = np.arange(lo, hi + bin_px, bin_px)
    centers_px = (edges[:-1] + edges[1:]) / 2
    centers_cm = centers_px * px_to_cm

    sum_g, _ = np.histogram(rel_perp, bins=edges, weights=gray_v)
    cnt_g, _ = np.histogram(rel_perp, bins=edges)
    mean_g = np.where(cnt_g > 0, sum_g / np.maximum(cnt_g, 1), 0)
    # 픽셀 없는 bin은 mean으로 채움
    valid = cnt_g > 0
    if valid.any():
        mean_g[~valid] = mean_g[valid].mean()

    # gray 반전 → 어두운 홈이 큰 값
    gray_dark = -mean_g  # 부호만 반전 (평균 0 근처 됨은 detrend 뒤)
    gray_dark_bp = bandpass_1d(gray_dark, SUB_BIN_CM,
                                BANDPASS_LO_CM, BANDPASS_HI_CM)
    gray_dark_norm = norm01(gray_dark_bp)

    # SAM 밀도
    in_band_sam = (np.abs(sam_perp_all - center) <= search_half_px) \
                  & (sam_ridge_all >= ridge_min) & (sam_ridge_all <= ridge_max)
    sam_rel = sam_perp_all[in_band_sam] - center
    sam_hist, _ = np.histogram(sam_rel, bins=edges)
    if sam_hist.max() > 0:
        sam_bp = bandpass_1d(sam_hist.astype(float), SUB_BIN_CM,
                              BANDPASS_LO_CM, BANDPASS_HI_CM)
        sam_norm = norm01(sam_bp)
    else:
        sam_norm = np.zeros_like(gray_dark_norm)

    combined = GRAY_INVERT_WEIGHT * gray_dark_norm + SAM_WEIGHT * sam_norm
    combined = combined / (GRAY_INVERT_WEIGHT + SAM_WEIGHT)

    # find_peaks
    min_dist_px = ROW_MIN_SPACING_CM / SUB_BIN_CM
    peaks, props = find_peaks(combined, distance=min_dist_px, height=0.15)
    # top-2 by height
    if len(peaks) > 2:
        heights = props["peak_heights"]
        order = np.argsort(heights)[::-1]
        peaks = np.sort(peaks[order[:2]])

    peaks_perp_px = centers_px[peaks]      # 상대 (두둑 중심 기준)
    peaks_abs_perp_px = peaks_perp_px + center  # 절대 (원점 기준)

    if len(peaks) == 2:
        gap_cm = float((peaks_perp_px[1] - peaks_perp_px[0]) * px_to_cm)
        status = "dual"
    elif len(peaks) == 1:
        gap_cm = np.nan
        status = "single"
    else:
        gap_cm = np.nan
        status = "none"

    return dict(
        perp_cm=centers_cm,
        gray_dark_norm=gray_dark_norm, sam_norm=sam_norm, combined=combined,
        peaks_local_perp_px=peaks_perp_px,
        peaks_abs_perp_px=peaks_abs_perp_px,
        gap_cm=gap_cm,
        n_leaves_in_band=int(in_band_sam.sum()),
        n_pixels_in_band=int(in_band.sum()),
        status=status,
    )


def classify_gap(gap_cm: float, n_peaks: int) -> str:
    if n_peaks < 2 or np.isnan(gap_cm):
        return "single"
    if NARROW_MIN_CM <= gap_cm <= NARROW_MAX_CM:
        return "narrow"
    if WIDE_MIN_CM <= gap_cm <= WIDE_MAX_CM:
        return "wide"
    return "other"


# ---- 시각화 ----
def visualize(res: dict, out_png: Path):
    setup_korean_font()
    fig = plt.figure(figsize=(24, 15))
    gs = fig.add_gridspec(3, 4, height_ratios=[2.5, 1, 1.2])

    rgb = res["rgb_disp"]; rgb_step = res["rgb_step"]
    origin = res["origin"]; perp = res["perp"]; ridge_dir = res["ridge_dir"]
    px_to_cm = res["px_to_cm"]
    ridges = res["ridges"]
    row_info = res["row_info"]

    # (1) 필지 전체 + 각 두둑 2개 파종 자국 라인
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.imshow(rgb)
    for r in ridges:
        info = row_info[r["ridge_id"]]
        rtype = info["type_final"]
        # 색: narrow=orange, wide=purple, single=gray, other=cyan
        cmap = {"narrow": "orange", "wide": "purple",
                 "single": "gray", "other": "cyan"}
        color = cmap.get(rtype, "gray")
        # 2 peaks 라인 그리기
        for perp_px in info["peaks_abs_perp_px"]:
            p1 = origin + r["ridge_min_px"] * ridge_dir + perp_px * perp
            p2 = origin + r["ridge_max_px"] * ridge_dir + perp_px * perp
            p1d = p1 / rgb_step; p2d = p2 / rgb_step
            ax1.plot([p1d[1], p2d[1]], [p1d[0], p2d[0]],
                     color=color, lw=0.4, alpha=0.85)
    ax1.set_title(f"Step 2 - 파종 줄 2개 (색상: 조간 유형)\n"
                  f"주황=좁은({NARROW_MIN_CM}~{NARROW_MAX_CM}cm), "
                  f"보라=넓은({WIDE_MIN_CM}~{WIDE_MAX_CM}cm), 회색=단일, 청록=벗어남",
                  fontsize=12, fontweight="bold")
    ax1.axis("off")

    # (2) 조간 분포 히스토그램
    ax2 = fig.add_subplot(gs[0, 2])
    gaps = [info["gap_cm"] for info in row_info.values()
            if not np.isnan(info["gap_cm"])]
    if len(gaps):
        ax2.hist(gaps, bins=np.arange(0, 100, 3), color="steelblue",
                 edgecolor="navy", alpha=0.8)
        ax2.axvspan(NARROW_MIN_CM, NARROW_MAX_CM, color="orange", alpha=0.15,
                    label=f"좁은 {NARROW_MIN_CM}~{NARROW_MAX_CM}cm")
        ax2.axvspan(WIDE_MIN_CM, WIDE_MAX_CM, color="purple", alpha=0.15,
                    label=f"넓은 {WIDE_MIN_CM}~{WIDE_MAX_CM}cm")
        ax2.axvline(np.median(gaps), color="red", lw=1.5,
                    label=f"median {np.median(gaps):.1f}cm")
        ax2.set_xlabel("조간 (cm)"); ax2.set_ylabel("두둑 수")
        ax2.set_title(f"조간 분포 (n={len(gaps)})", fontsize=12)
        ax2.legend(fontsize=9)
        ax2.grid(alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "조간 데이터 없음", ha="center",
                 va="center", transform=ax2.transAxes)
        ax2.axis("off")

    # (3) 유형 통계 텍스트
    ax3 = fig.add_subplot(gs[0, 3])
    ax3.axis("off")
    types = [info["type_final"] for info in row_info.values()]
    n_narrow = sum(t == "narrow" for t in types)
    n_wide = sum(t == "wide" for t in types)
    n_single = sum(t == "single" for t in types)
    n_other = sum(t == "other" for t in types)
    narrow_gaps = [info["gap_cm"] for info in row_info.values()
                    if info["type_final"] == "narrow"]
    wide_gaps = [info["gap_cm"] for info in row_info.values()
                  if info["type_final"] == "wide"]
    lines = [
        f"필지: {res['field']}",
        f"두둑 총 {len(ridges)}개",
        "",
        f"- 좁은 두둑 (narrow): {n_narrow}",
        f"    조간 median: "
        + (f"{np.median(narrow_gaps):.1f}cm" if narrow_gaps else "N/A"),
        f"- 넓은 두둑 (wide):   {n_wide}",
        f"    조간 median: "
        + (f"{np.median(wide_gaps):.1f}cm" if wide_gaps else "N/A"),
        f"- 단일 (peak 1 or 0): {n_single}",
        f"- 벗어남 (other):     {n_other}",
        "",
        f"검증: n_narrow + n_wide = "
        f"{n_narrow + n_wide} / {len(ridges)} "
        f"({(n_narrow+n_wide)/len(ridges)*100:.1f}%)",
    ]
    ax3.text(0.02, 0.98, "\n".join(lines), fontsize=12,
             va="top", ha="left", transform=ax3.transAxes)

    # (4) 예시 두둑 sub-profile (n_narrow와 n_wide 중 각 2개씩)
    example_ridges = []
    for target_type in ["narrow", "wide", "single", "other"]:
        for rid, info in row_info.items():
            if info["type_final"] == target_type:
                example_ridges.append((rid, info))
                break
    example_ridges = example_ridges[:4]

    for i, (rid, info) in enumerate(example_ridges):
        ax = fig.add_subplot(gs[1 + i // 2, (i % 2) * 2 : (i % 2) * 2 + 2])
        if info["status"] == "empty":
            ax.text(0.5, 0.5, f"두둑 {rid}: 데이터 없음",
                    ha="center", va="center", transform=ax.transAxes)
            continue
        ax.plot(info["perp_cm"], info["gray_dark_norm"], color="C1",
                alpha=0.7, label="Gray 반전 (norm)")
        ax.plot(info["perp_cm"], info["sam_norm"], color="C2",
                alpha=0.7, label="SAM 밀도 (norm)")
        ax.plot(info["perp_cm"], info["combined"], color="C0",
                lw=1.5, label="결합")
        for p_cm in info["peaks_local_perp_px"] * px_to_cm:
            ax.axvline(p_cm, color="red", lw=1.0, alpha=0.7)
        gap = info["gap_cm"]
        gap_str = f"조간 {gap:.1f}cm" if not np.isnan(gap) else "조간 N/A"
        ax.set_title(f"두둑 {rid} ({info['type_final']}) - {gap_str}, "
                     f"SAM {info['n_leaves_in_band']}잎",
                     fontsize=11)
        ax.set_xlabel("두둑 중심 기준 perp (cm)")
        ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.3)

    plt.suptitle(f"Step 2: 두둑별 dual-row 검출 + 조간 - {res['field']}",
                 fontsize=15, fontweight="bold", y=1.005)
    plt.tight_layout()
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close()


# ---- 메인 ----
def main():
    if len(sys.argv) < 2:
        print("사용: python step2_rows.py <field_name>")
        sys.exit(1)
    field = sys.argv[1]
    t0 = time.time()

    print(f"[{field}] npz 로드")
    d = load_field_npz(field)
    sam_pts = leaf_centroids(d["leaves"])

    field_dir = get_field_dir(field)
    r_npz_path = field_dir / "ridges.npz"
    if not r_npz_path.exists():
        print(f"먼저 step1_ridges 실행 필요: {r_npz_path}")
        sys.exit(1)
    r_npz = np.load(r_npz_path, allow_pickle=True)
    origin = r_npz["origin"]
    perp = r_npz["perp"]
    ridge_dir = r_npz["ridge_dir"]
    px_to_cm = float(r_npz["px_to_cm"][0])
    ridge_arr = r_npz["ridge_arr"]
    ridge_types = r_npz["ridge_types"]
    print(f"   두둑 {len(ridge_arr)}개, 방향축 px_to_cm={px_to_cm:.4f}")

    ridges = []
    for i, row in enumerate(ridge_arr):
        ridges.append(dict(
            ridge_id=int(row[0]), n_rows=int(row[1]),
            center_perp_px=float(row[2]), width_px=float(row[3]),
            width_cm=float(row[4]), band_half_cm=float(row[5]),
            ridge_min_px=float(row[6]), ridge_max_px=float(row[7]),
            length_cm=float(row[8]), n_leaves=int(row[9]),
            ridge_type=str(ridge_types[i]),
        ))

    print(f"[{field}] Gray/mask 준비")
    gray, mask = rgb_to_gray_and_mask(d["rgb_disp"])
    H, W = gray.shape
    ys_d, xs_d = np.mgrid[0:H, 0:W]
    coords = np.stack([(ys_d * d["rgb_step"]).astype(np.float32).ravel(),
                        (xs_d * d["rgb_step"]).astype(np.float32).ravel()], axis=1)
    gray_vals = gray.ravel()
    gray_mask_flat = mask.ravel()
    coords_c = coords - origin  # 원점 기준

    sam_rel = sam_pts - origin
    sam_perp_all = sam_rel @ perp
    sam_ridge_all = sam_rel @ ridge_dir

    print(f"[{field}] 두둑별 sub-scan")
    row_info = {}
    for i, r in enumerate(ridges):
        info = compute_ridge_sub_profile(
            r, origin, perp, ridge_dir, px_to_cm,
            coords_c, gray_vals, gray_mask_flat,
            sam_perp_all, sam_ridge_all,
        )
        info["type_final"] = classify_gap(info.get("gap_cm", np.nan),
                                           len(info.get("peaks_abs_perp_px", [])))
        row_info[r["ridge_id"]] = info

    n_narrow = sum(v["type_final"] == "narrow" for v in row_info.values())
    n_wide = sum(v["type_final"] == "wide" for v in row_info.values())
    n_single = sum(v["type_final"] == "single" for v in row_info.values())
    n_other = sum(v["type_final"] == "other" for v in row_info.values())
    narrow_gaps = [v["gap_cm"] for v in row_info.values() if v["type_final"] == "narrow"]
    wide_gaps = [v["gap_cm"] for v in row_info.values() if v["type_final"] == "wide"]
    print(f"   두둑 {len(ridges)}개 - narrow {n_narrow}, wide {n_wide}, "
          f"single {n_single}, other {n_other}")
    if narrow_gaps:
        print(f"   좁은 조간 median {np.median(narrow_gaps):.1f}cm "
              f"(n={len(narrow_gaps)})")
    if wide_gaps:
        print(f"   넓은 조간 median {np.median(wide_gaps):.1f}cm "
              f"(n={len(wide_gaps)})")

    out_npz = field_dir / "rows.npz"
    out_png = field_dir / "step2_rows.png"

    # 저장: 두둑당 row info
    row_arr = np.array([[
        rid,
        len(info.get("peaks_abs_perp_px", [])),
        info["peaks_abs_perp_px"][0] if len(info.get("peaks_abs_perp_px", [])) >= 1 else np.nan,
        info["peaks_abs_perp_px"][1] if len(info.get("peaks_abs_perp_px", [])) >= 2 else np.nan,
        info.get("gap_cm", np.nan),
        info.get("n_leaves_in_band", 0),
    ] for rid, info in sorted(row_info.items())], dtype=np.float32)
    type_arr = np.array([info["type_final"] for _, info
                          in sorted(row_info.items())])

    np.savez_compressed(
        out_npz,
        row_arr=row_arr,        # [ridge_id, n_peaks, p1_px, p2_px, gap_cm, n_leaves]
        type_arr=type_arr,
    )
    print(f"[{field}] 저장: {out_npz.name}")

    visualize({
        "field": field, "rgb_disp": d["rgb_disp"], "rgb_step": d["rgb_step"],
        "origin": origin, "perp": perp, "ridge_dir": ridge_dir,
        "px_to_cm": px_to_cm, "ridges": ridges, "row_info": row_info,
    }, out_png)
    print(f"[{field}] 저장: {out_png.name}")
    print(f"[{field}] 완료 ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
