"""
두둑 너비·줄 간격 측정 근거 시각화.

각 필지에 대해:
  1. 전체 RGB + 검출된 줄 라인 (색깔로 gap 종류 구분)
  2. 확대 뷰 (10m × 10m) — 라인 + 실제 측정값 라벨
  3. 확대 뷰 (2m × 2m) — 극세 확인용
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import KMeans


ROOT = Path(r"C:/Users/user/Desktop/분석프로젝트/Soybean_Emergence_Rate")
OUT_DIR = ROOT / "result" / "sam_test"
VIZ_DIR = OUT_DIR / "final_report"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

FIELDS = {
    "Smart":  dict(name="스마트 파종기", short="Smart",
                    npz="GJSM-1-1_Smart_sam_FULL_v4.npz",
                    spec=dict(ridge_width=70, row_gap=30, plant_gap=20, furrow=35),
                    min_ridge_dist_cm=15, smooth_sigma=0.4,
                    color="#0066cc"),
    "Normal": dict(name="일반 파종기", short="Normal",
                    npz="GJSM-1-1_normal_sam_FULL_v4.npz",
                    spec=dict(ridge_width=140, row_gap=70, plant_gap=20, furrow=35),
                    min_ridge_dist_cm=55, smooth_sigma=1.2,
                    color="#cc6600"),
}


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


def analyze(key):
    field = FIELDS[key]
    d = np.load(OUT_DIR / field["npz"])
    leaves = d["leaf_arr"]
    gsd = float(d["gsd_ds"][0])
    rgb_disp = d["rgb_disp"]
    step = int(d["rgb_step"][0])

    centroids = leaves[:, :2]
    px_to_cm = gsd * 100

    mean = centroids.mean(axis=0)
    centered = centroids - mean
    ang = find_best_angle(centered, gsd)
    a = np.radians(ang)
    perp = np.array([np.cos(a), np.sin(a)])
    par = np.array([-np.sin(a), np.cos(a)])
    proj_perp = centered @ perp
    proj_par = centered @ par

    bin_size_px = 0.5 / px_to_cm
    edges = np.arange(proj_perp.min(), proj_perp.max() + bin_size_px, bin_size_px)
    hist, _ = np.histogram(proj_perp, bins=edges)
    hist_smooth = gaussian_filter1d(hist.astype(float), sigma=field["smooth_sigma"])
    peaks, _ = find_peaks(hist_smooth,
                          distance=field["min_ridge_dist_cm"] / px_to_cm,
                          height=hist_smooth.max() * 0.02)
    peak_pos_px = edges[peaks] + bin_size_px / 2
    peak_pos_cm = peak_pos_px * px_to_cm
    gaps_cm = np.diff(peak_pos_cm)

    # KMeans 로 작은/큰 간격 분류
    spec = field["spec"]
    expect_period = spec["ridge_width"] + spec["furrow"]
    if len(gaps_cm) >= 2:
        init = np.array([[spec["row_gap"]], [expect_period - spec["row_gap"]]])
        km = KMeans(n_clusters=2, n_init=1, init=init, random_state=0).fit(gaps_cm.reshape(-1, 1))
        labels = km.labels_
        centers = km.cluster_centers_.flatten()
        order = np.argsort(centers)
        small_label = order[0]
        # 각 gap의 종류 (0=작은=내부줄, 1=큰=두둑사이)
        gap_type = np.where(labels == small_label, 0, 1)
        small_median = float(np.median(gaps_cm[gap_type == 0]))
        large_median = float(np.median(gaps_cm[gap_type == 1]))
    else:
        gap_type = np.zeros(len(gaps_cm), dtype=int)
        small_median = np.nan
        large_median = np.nan

    return dict(
        field=field,
        rgb_disp=rgb_disp, step=step,
        centroids=centroids, mean=mean,
        perp=perp, par=par, angle=ang,
        peak_pos_px=peak_pos_px, peak_pos_cm=peak_pos_cm,
        gaps_cm=gaps_cm, gap_type=gap_type,
        small_median=small_median, large_median=large_median,
        px_to_cm=px_to_cm,
        H_px=rgb_disp.shape[0] * step, W_px=rgb_disp.shape[1] * step,
    )


def draw_lines_on_rgb(ax, res, offset_perp_cm=0, offset_par_cm=0,
                       label_lines=False, extent_cm=None):
    """
    ax 위 RGB에 검출된 줄 라인 그리기.
    offset_perp/par_cm: 확대 뷰의 원점 이동값 (cm 단위)
    label_lines: True면 간격 값을 라벨링
    extent_cm: (perp_min, perp_max, par_min, par_max) → 라인 그리는 범위
    """
    step = res["step"]
    mean = res["mean"]
    perp = res["perp"]; par = res["par"]
    px_to_cm = res["px_to_cm"]

    for i, (pk_cm, gap_i) in enumerate(zip(res["peak_pos_cm"],
                                            list(res["gap_type"]) + [None])):
        # 이 봉우리 앞에 있는 gap의 종류로 색상 결정
        prev_gap_type = res["gap_type"][i - 1] if i > 0 else None
        if prev_gap_type is None:
            color = "orange"; lw = 0.5
        elif prev_gap_type == 0:
            color = "#ff4444"; lw = 0.9  # 작은 gap (내부 줄)
        else:
            color = "#22cc22"; lw = 0.9  # 큰 gap (두둑 사이)

        # 라인은 par 방향으로 무한히 뻗음
        pk_px = pk_cm / px_to_cm
        # 화면 좌표계로 변환
        # 점 = mean + pk_px * perp + t * par
        ts = np.linspace(-20000, 20000, 2)
        line_y = (mean[0] + pk_px * perp[0] + ts * par[0]) / step
        line_x = (mean[1] + pk_px * perp[1] + ts * par[1]) / step
        ax.plot(line_x, line_y, color=color, lw=lw, alpha=0.7)

    ax.set_xlim(0, res["rgb_disp"].shape[1])
    ax.set_ylim(res["rgb_disp"].shape[0], 0)


def draw_full(res, out_png):
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(20, 12))

    # 전체 뷰
    ax = axes[0]
    ax.imshow(res["rgb_disp"])
    draw_lines_on_rgb(ax, res)
    ax.set_title(f"{res['field']['name']} 전체 필지 (전체 라인 표시)",
                  fontsize=13, fontweight="bold")
    ax.axis("off")

    # 확대 뷰 — 필지 중심부 20m × 20m
    ax = axes[1]
    ax.imshow(res["rgb_disp"])
    step = res["step"]
    # 중심 근처 20m × 20m 확대 (표시용 축소된 좌표계에서)
    H, W = res["rgb_disp"].shape[:2]
    px_to_cm = res["px_to_cm"]
    m_per_disp_px = step * px_to_cm / 100  # 하나의 표시 px = ? m
    zoom_size_m = 15
    zoom_size_disp_px = int(zoom_size_m / m_per_disp_px)
    cy, cx = H // 2, W // 2
    x0, x1 = max(0, cx - zoom_size_disp_px // 2), min(W, cx + zoom_size_disp_px // 2)
    y0, y1 = max(0, cy - zoom_size_disp_px // 2), min(H, cy + zoom_size_disp_px // 2)
    draw_lines_on_rgb(ax, res)
    ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
    ax.set_title(f"확대 뷰 (약 {zoom_size_m}m × {zoom_size_m}m)",
                  fontsize=13, fontweight="bold")
    ax.axis("off")

    # 색상 범례
    legend_els = [
        mpatches.Patch(color="#ff4444",
                        label=f"작은 gap ({res['small_median']:.1f}cm median) = 내부 줄 간격"),
        mpatches.Patch(color="#22cc22",
                        label=f"큰 gap ({res['large_median']:.1f}cm median) = 두둑 사이"),
    ]
    fig.legend(handles=legend_els, loc="upper center",
                bbox_to_anchor=(0.5, 0.97), ncol=2, fontsize=11)

    fig.suptitle(f"{res['field']['name']} — 두둑·줄 검출 라인 시각화\n"
                 f"[스펙: 두둑 {res['field']['spec']['ridge_width']}cm, "
                 f"내부 줄 {res['field']['spec']['row_gap']}cm, "
                 f"고랑 {res['field']['spec']['furrow']}cm]",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close()


def draw_measurement_diagram(res, out_png):
    """밀도 프로파일 위에 각 gap을 색상+거리 라벨로 시각화."""
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(1, 1, figsize=(24, 6))

    peak_pos_cm = res["peak_pos_cm"]
    gaps_cm = res["gaps_cm"]
    gap_type = res["gap_type"]

    # 각 봉우리 세로선 + 라벨
    y_top = 1.0
    for i in range(len(peak_pos_cm) - 1):
        p0 = peak_pos_cm[i]; p1 = peak_pos_cm[i + 1]
        color = "#ff4444" if gap_type[i] == 0 else "#22cc22"
        # 두 봉우리를 연결하는 수평 화살표
        y = y_top - 0.15
        ax.annotate("", xy=(p1, y), xytext=(p0, y),
                     arrowprops=dict(arrowstyle="<->", color=color, lw=1.2))
        # 중간에 거리 라벨
        gap = gaps_cm[i]
        ax.text((p0 + p1) / 2, y - 0.08, f"{gap:.0f}",
                 ha="center", va="top", fontsize=7, color=color,
                 rotation=90)

    # 봉우리 위치 세로선
    for p in peak_pos_cm:
        ax.axvline(p, color="gray", lw=0.3, alpha=0.5, ymax=0.7)

    ax.set_xlim(peak_pos_cm.min() - 50, peak_pos_cm.min() + 2000)  # 첫 20m만 표시
    ax.set_ylim(-0.2, 1.2)
    ax.set_xlabel("두둑 수직축 위치 (cm)")
    ax.set_yticks([])
    ax.set_title(f"{res['field']['name']} — 봉우리 간격 측정 시각화 (첫 20m)\n"
                 f"빨강 = 내부 줄 간격 (median {res['small_median']:.1f}cm), "
                 f"초록 = 두둑 사이 (median {res['large_median']:.1f}cm)  "
                 f"[스펙 내부 {res['field']['spec']['row_gap']}cm]",
                 fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close()


def main():
    print("=" * 60)
    print("두둑·줄 측정 근거 시각화")
    print("=" * 60)
    for key in FIELDS:
        print(f"\n[{key}]")
        res = analyze(key)
        print(f"  봉우리 {len(res['peak_pos_cm'])}개, "
              f"작은 gap {res['small_median']:.1f}cm, "
              f"큰 gap {res['large_median']:.1f}cm")

        draw_full(res, VIZ_DIR / f"4_{key}_line_overlay.png")
        print(f"  저장: 4_{key}_line_overlay.png")

        draw_measurement_diagram(res, VIZ_DIR / f"5_{key}_gap_diagram.png")
        print(f"  저장: 5_{key}_gap_diagram.png")

    print(f"\n✅ 완료: {VIZ_DIR}")


if __name__ == "__main__":
    main()
