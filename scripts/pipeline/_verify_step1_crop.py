"""
Step 1 검증 - 필지 rgb_disp 코너/중앙 crop 3장에 두둑 폴리곤·행 라인 overlay.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.common import load_field_npz, get_field_dir, setup_korean_font


def crop_and_draw(rgb, rgb_step, origin, perp, ridge_dir, px_to_cm,
                    ridges, y0, x0, sz, ax):
    """(y0,x0) ~ (y0+sz, x0+sz) 영역 crop 후 두둑 폴리곤 오버레이."""
    crop = rgb[y0:y0+sz, x0:x0+sz]
    ax.imshow(crop)

    for r in ridges:
        band_half_px = r["band_half_cm"] / px_to_cm
        # 코너 4점 (원본 disp 좌표)
        corners = []
        for r_off, p_off in [
            (r["ridge_min_px"], -band_half_px),
            (r["ridge_max_px"], -band_half_px),
            (r["ridge_max_px"], +band_half_px),
            (r["ridge_min_px"], +band_half_px),
        ]:
            pt = origin + r_off * ridge_dir + (r["center_perp_px"] + p_off) * perp
            corners.append(pt / rgb_step)
        corners = np.array(corners)
        # crop 좌표로 변환 (y - y0, x - x0)
        crop_corners = corners - np.array([y0, x0])
        # xy = (x, y) 순
        poly = mpatches.Polygon(crop_corners[:, ::-1], closed=True,
                                 edgecolor="cyan", facecolor="none", linewidth=1.5)
        ax.add_patch(poly)

        # 행 중심선
        p1 = origin + r["ridge_min_px"] * ridge_dir + r["center_perp_px"] * perp
        p2 = origin + r["ridge_max_px"] * ridge_dir + r["center_perp_px"] * perp
        p1d = p1 / rgb_step - np.array([y0, x0])
        p2d = p2 / rgb_step - np.array([y0, x0])
        ax.plot([p1d[1], p2d[1]], [p1d[0], p2d[0]],
                color="red", lw=1.0, alpha=0.9)
        # ridge id 텍스트 (중심)
        cy = (p1d[0] + p2d[0]) / 2
        cx = (p1d[1] + p2d[1]) / 2
        if 0 <= cy <= sz and 0 <= cx <= sz:
            ax.text(cx, cy, str(r["ridge_id"]), color="yellow", fontsize=9,
                     fontweight="bold", ha="center", va="center",
                     bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.6))

    ax.set_xlim(0, sz); ax.set_ylim(sz, 0)
    ax.axis("off")


def main():
    field = sys.argv[1] if len(sys.argv) > 1 else "GJSM-1-1_Smart"
    setup_korean_font()

    d = load_field_npz(field)
    rgb = d["rgb_disp"]; rgb_step = d["rgb_step"]

    field_dir = get_field_dir(field)
    r_npz = np.load(field_dir / "ridges.npz", allow_pickle=True)
    origin = r_npz["origin"]
    perp = r_npz["perp"]
    ridge_dir = r_npz["ridge_dir"]
    px_to_cm = float(r_npz["px_to_cm"][0])
    ridge_arr = r_npz["ridge_arr"]
    ridge_types = r_npz["ridge_types"]

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

    H, W, _ = rgb.shape
    sz = 400   # 400 disp px = 21m 시야
    # 필지 mask(비검정) center of mass 기준으로 crop 위치 결정
    mask = (rgb.astype(np.int32).sum(axis=2) > 5)
    ys, xs = np.where(mask)
    y_lo, y_hi = ys.min(), ys.max()
    x_lo, x_hi = xs.min(), xs.max()
    ymid, xmid = (y_lo + y_hi) // 2, (x_lo + x_hi) // 2
    positions = [
        ("좌상 (필지 안)", ymid - sz - 50, xmid - sz - 50),
        ("중앙",           ymid - sz//2,    xmid - sz//2),
        ("우하 (필지 안)", ymid + 50,       xmid + 50),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(28, 10))
    for ax, (label, y, x) in zip(axes, positions):
        y = max(0, min(y, H - sz))
        x = max(0, min(x, W - sz))
        crop_and_draw(rgb, rgb_step, origin, perp, ridge_dir, px_to_cm,
                       ridges, y, x, sz, ax)
        ax.set_title(f"{label}  (y={y}, x={x}, {sz*rgb_step*0.01:.1f}m 시야)",
                     fontsize=13, fontweight="bold")

    plt.suptitle(f"Step 1 검증 - {field}  "
                  f"cyan=두둑 밴드, red=중심선, yellow=ridge_id",
                  fontsize=15, fontweight="bold", y=1.02)
    out = field_dir / "_verify_step1_crops.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
