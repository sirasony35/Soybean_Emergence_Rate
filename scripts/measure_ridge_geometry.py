"""
두둑 기하 측정 — 이미지에서 두둑 폭·고랑 폭·반복 주기 자동 산출.

원리:
  dual-row 파종에서 밀도 프로파일의 인접 봉우리 간격은 번갈아 나타남:
    - 작은 간격 = 두둑 안 두 줄 간격 (내부)
    - 큰 간격 = 두둑 사이 줄-줄 간격 (고랑 포함)
  → 반복 주기 = 작은 + 큰,  고랑 폭 = 반복 주기 − 두둑 폭

봉우리 검출 완화:
  - RIDGE_SPACING_MIN_CM 15 (기존 20 → 15)
  - PEAK_HEIGHT_FRAC 0.02 (기존 0.03 → 0.02)

사용:
  python -u scripts/measure_ridge_geometry.py
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import KMeans


ROOT = Path(r"C:/Users/user/Desktop/분석프로젝트/Soybean_Emergence_Rate")
OUT_DIR = ROOT / "result" / "sam_test"
GEO_DIR = OUT_DIR / "geometry"
GEO_DIR.mkdir(parents=True, exist_ok=True)

# 스펙 (사용자 확정)
FIELDS_SPEC = {
    "GJSM-1-1_Smart":  dict(ridge_width_cm=70,  inner_row_gap_cm=30, furrow_cm_spec=35),
    "GJSM-1-1_normal": dict(ridge_width_cm=140, inner_row_gap_cm=70, furrow_cm_spec=35),
}

# 완화된 봉우리 검출
RIDGE_SPACING_MIN_CM = 15
HIST_BIN_CM = 0.5
HIST_SMOOTH_SIGMA = 0.4
PEAK_HEIGHT_FRAC = 0.02


def find_best_angle(centered, gsd_m, step_deg=1.0):
    px_to_cm = gsd_m * 100
    bin_size_px = 1.0 / px_to_cm
    angles = np.arange(0, 180, step_deg)
    scores = np.zeros(len(angles))
    for i, a_deg in enumerate(angles):
        a = np.radians(a_deg)
        perp = np.array([np.cos(a), np.sin(a)])
        proj_a = centered @ perp
        edges = np.arange(proj_a.min(), proj_a.max() + bin_size_px, bin_size_px)
        hist, _ = np.histogram(proj_a, bins=edges)
        hist_smooth = gaussian_filter1d(hist.astype(float), sigma=1.5)
        m = hist_smooth.mean() + 1e-9
        scores[i] = hist_smooth.var() / (m ** 2)
    return angles[int(np.argmax(scores))]


def compute_profile_and_peaks(centroids, gsd_m):
    px_to_cm = gsd_m * 100
    mean = centroids.mean(axis=0)
    centered = centroids - mean
    ridge_angle = find_best_angle(centered, gsd_m)
    a = np.radians(ridge_angle)
    perp_dir = np.array([np.cos(a), np.sin(a)])
    proj = centered @ perp_dir
    bin_size_px = HIST_BIN_CM / px_to_cm
    edges = np.arange(proj.min(), proj.max() + bin_size_px, bin_size_px)
    hist, _ = np.histogram(proj, bins=edges)
    hist_smooth = gaussian_filter1d(hist.astype(float), sigma=HIST_SMOOTH_SIGMA)
    min_dist_px = RIDGE_SPACING_MIN_CM / px_to_cm
    peaks, _ = find_peaks(hist_smooth,
                          distance=min_dist_px,
                          height=hist_smooth.max() * PEAK_HEIGHT_FRAC)
    peak_pos_px = edges[peaks] + bin_size_px / 2
    peak_pos_cm = peak_pos_px * px_to_cm
    return dict(
        ridge_angle_deg=float(ridge_angle),
        peak_pos_cm=peak_pos_cm,
        centers_cm=(edges[:-1] + edges[1:]) / 2 * px_to_cm,
        hist=hist, hist_smooth=hist_smooth,
        px_to_cm=px_to_cm,
    )


def classify_gaps(gaps_cm: np.ndarray, expected_small: float, expected_large: float):
    """KMeans(k=2) 로 봉우리 간격 클러스터링."""
    if len(gaps_cm) < 2:
        return None
    X = gaps_cm.reshape(-1, 1)
    # 초기 중심 = 스펙 예상값
    init = np.array([[expected_small], [expected_large]])
    km = KMeans(n_clusters=2, n_init=1, init=init, random_state=0).fit(X)
    labels = km.labels_
    centers = km.cluster_centers_.flatten()
    # 작은 쪽 = 0, 큰 쪽 = 1
    order = np.argsort(centers)
    small_label = order[0]; large_label = order[1]
    small_mask = labels == small_label
    large_mask = labels == large_label
    return dict(
        small_gaps=gaps_cm[small_mask],
        large_gaps=gaps_cm[large_mask],
        small_center=float(centers[small_label]),
        large_center=float(centers[large_label]),
    )


def analyze(field: str, spec: dict):
    npz_path = OUT_DIR / f"{field}_sam_FULL_v4.npz"
    d = np.load(npz_path)
    leaves = d["leaf_arr"]
    gsd = float(d["gsd_ds"][0])
    centroids = leaves[:, :2]

    prof = compute_profile_and_peaks(centroids, gsd)
    peaks = prof["peak_pos_cm"]
    gaps = np.diff(peaks) if len(peaks) > 1 else np.array([])

    # 스펙 예상 큰 간격 = 반복주기 - 내부 간격
    period_spec = spec["ridge_width_cm"] + spec["furrow_cm_spec"]
    large_expected = period_spec - spec["inner_row_gap_cm"]
    clust = classify_gaps(gaps, spec["inner_row_gap_cm"], large_expected)

    result = dict(
        field=field,
        spec=spec,
        n_peaks=int(len(peaks)),
        n_gaps=int(len(gaps)),
        gap_median=float(np.median(gaps)) if len(gaps) else np.nan,
        gap_mean=float(gaps.mean()) if len(gaps) else np.nan,
        ridge_angle_deg=prof["ridge_angle_deg"],
        peaks_cm=peaks.tolist(),
        gaps_cm=gaps.tolist(),
        _profile=prof,
    )
    if clust is not None:
        small_med = float(np.median(clust["small_gaps"]))
        large_med = float(np.median(clust["large_gaps"]))
        period_meas = small_med + large_med
        # 두둑 폭 = 반복 주기 − 고랑
        furrow_meas = period_meas - spec["ridge_width_cm"]
        result.update(dict(
            small_gap_median_cm=small_med,
            large_gap_median_cm=large_med,
            small_gap_n=int(len(clust["small_gaps"])),
            large_gap_n=int(len(clust["large_gaps"])),
            period_measured_cm=period_meas,
            furrow_measured_cm=furrow_meas,
            period_spec_cm=period_spec,
            furrow_spec_cm=spec["furrow_cm_spec"],
        ))
    return result


def draw_profile(res: dict, out_png: Path):
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    fig, (ax_p, ax_h) = plt.subplots(2, 1, figsize=(18, 10),
                                       gridspec_kw={"height_ratios": [1.4, 1]})
    prof = res["_profile"]
    ax_p.plot(prof["centers_cm"], prof["hist_smooth"], color="C0", lw=1.2,
              label="smoothed 밀도")
    # 봉우리 색상 (스몰/라지 클래스별)
    peaks = np.array(res["peaks_cm"])
    if "small_gap_median_cm" in res:
        gaps = np.array(res["gaps_cm"])
        small_med = res["small_gap_median_cm"]; large_med = res["large_gap_median_cm"]
        threshold = (small_med + large_med) / 2
        # 각 봉우리 마다 다음 gap의 종류로 라벨
        colors = []
        for i in range(len(peaks)):
            if i < len(gaps):
                colors.append("red" if gaps[i] <= threshold else "green")
            else:
                colors.append("gray")
        for i, (p, c) in enumerate(zip(peaks, colors)):
            ax_p.axvline(p, color=c, lw=0.8, alpha=0.5)
    else:
        for p in peaks:
            ax_p.axvline(p, color="orange", lw=0.8, alpha=0.5)
    ax_p.set_xlabel("두둑 수직축 (cm)")
    ax_p.set_ylabel("잎 개수")
    title = f"{res['field']}  ·  봉우리 {res['n_peaks']}개  ·  각도 {res['ridge_angle_deg']:.0f}°"
    ax_p.set_title(title, fontsize=14, fontweight="bold")
    ax_p.grid(alpha=0.3); ax_p.legend()

    # 간격 히스토그램
    gaps = np.array(res["gaps_cm"])
    if len(gaps):
        bins = np.linspace(0, min(gaps.max() + 10, 200), 60)
        ax_h.hist(gaps, bins=bins, color="lightblue", edgecolor="C0")
        if "small_gap_median_cm" in res:
            ax_h.axvline(res["small_gap_median_cm"], color="red", lw=2,
                          label=f"작은 간격 median = {res['small_gap_median_cm']:.1f}cm "
                                f"(내부 줄, 스펙 {res['spec']['inner_row_gap_cm']}cm)")
            ax_h.axvline(res["large_gap_median_cm"], color="green", lw=2,
                          label=f"큰 간격 median = {res['large_gap_median_cm']:.1f}cm "
                                f"(두둑 사이)")
        ax_h.axvline(res["spec"]["inner_row_gap_cm"], color="red", lw=1, ls="--",
                      alpha=0.5, label=f"스펙 내부 {res['spec']['inner_row_gap_cm']}cm")
        ax_h.set_xlabel("인접 봉우리 간격 (cm)")
        ax_h.set_ylabel("빈도")
        ax_h.set_title("봉우리 간격 분포 (bimodal: 내부 vs 두둑 사이)",
                        fontsize=13, fontweight="bold")
        ax_h.legend(fontsize=10)
        ax_h.grid(alpha=0.3)

    if "period_measured_cm" in res:
        info = (f"반복 주기 실측 {res['period_measured_cm']:.1f}cm "
                f"(스펙 {res['period_spec_cm']}cm)\n"
                f"고랑 폭 실측 {res['furrow_measured_cm']:.1f}cm "
                f"(스펙 {res['furrow_spec_cm']}cm)")
        fig.text(0.5, 1.01, info, ha="center", fontsize=13,
                 fontweight="bold", color="#004488")

    plt.tight_layout()
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close()


def make_markdown(results: list) -> str:
    L = []
    L.append("# 두둑 기하 이미지 측정 vs 스펙 비교\n")
    L.append("**원리**: 밀도 프로파일 봉우리 간격을 KMeans(k=2)로 분류 → "
             "작은 간격(내부 줄) + 큰 간격(두둑 사이) → 반복 주기·고랑 폭 산출\n")

    L.append("## 필지별 측정 결과\n")
    L.append("| 필지 | 봉우리수 | 작은 간격(내부 줄) | 큰 간격(두둑 사이) | "
             "반복 주기 | 고랑 폭 |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for r in results:
        if "period_measured_cm" not in r:
            L.append(f"| {r['field']} | {r['n_peaks']} | - | - | - | - |")
            continue
        L.append(
            f"| {r['field']} | {r['n_peaks']} | "
            f"{r['small_gap_median_cm']:.1f} cm "
            f"(스펙 {r['spec']['inner_row_gap_cm']}) | "
            f"{r['large_gap_median_cm']:.1f} cm | "
            f"{r['period_measured_cm']:.1f} cm "
            f"(스펙 {r['period_spec_cm']}) | "
            f"**{r['furrow_measured_cm']:.1f} cm** "
            f"(스펙 {r['furrow_spec_cm']}) |")

    L.append("\n## 해석\n")
    for r in results:
        if "period_measured_cm" not in r:
            continue
        L.append(f"### {r['field']}")
        s = r["spec"]
        L.append(f"- 스펙: 두둑 폭 {s['ridge_width_cm']}cm, "
                 f"내부 줄 {s['inner_row_gap_cm']}cm, 고랑 {s['furrow_cm_spec']}cm")
        L.append(f"- 실측: 내부 줄 **{r['small_gap_median_cm']:.1f}cm**, "
                 f"고랑 폭 **{r['furrow_measured_cm']:.1f}cm**")
        err_inner = abs(r["small_gap_median_cm"] - s["inner_row_gap_cm"])
        err_furrow = abs(r["furrow_measured_cm"] - s["furrow_cm_spec"])
        L.append(f"- 오차: 내부 줄 ±{err_inner:.1f}cm, 고랑 ±{err_furrow:.1f}cm\n")
    return "\n".join(L)


def main():
    print("=" * 60)
    print("두둑 기하 측정 — 이미지 vs 스펙")
    print("=" * 60)

    results = []
    for field, spec in FIELDS_SPEC.items():
        print(f"\n[{field}]")
        r = analyze(field, spec)
        results.append(r)
        print(f"  봉우리 {r['n_peaks']}개, "
              f"간격 median {r['gap_median']:.1f}cm  mean {r['gap_mean']:.1f}cm")
        if "period_measured_cm" in r:
            print(f"  작은 간격 median: {r['small_gap_median_cm']:.1f}cm "
                  f"(스펙 {spec['inner_row_gap_cm']}cm)")
            print(f"  큰 간격 median : {r['large_gap_median_cm']:.1f}cm")
            print(f"  반복 주기 실측 : {r['period_measured_cm']:.1f}cm "
                  f"(스펙 {r['period_spec_cm']}cm)")
            print(f"  고랑 폭 실측  : {r['furrow_measured_cm']:.1f}cm "
                  f"(스펙 {spec['furrow_cm_spec']}cm)")
        draw_profile(r, GEO_DIR / f"{field}_geometry.png")

    md = make_markdown(results)
    (GEO_DIR / "summary.md").write_text(md, encoding="utf-8")
    print("\n" + md)

    def clean(r):
        return {k: v for k, v in r.items() if not k.startswith("_")}
    (GEO_DIR / "summary.json").write_text(
        json.dumps([clean(r) for r in results], ensure_ascii=False, indent=2),
        encoding="utf-8")

    print(f"\n📁 저장: {GEO_DIR}")


if __name__ == "__main__":
    main()
