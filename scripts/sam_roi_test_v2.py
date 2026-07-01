"""
SAM ROI 테스트 v2 — 검출 강화 (recall 우선)

v1(_v3) 대비 변경:
  - points_per_side: 48 → 64       (격자 1.78배 조밀)
  - pred_iou_thresh: 0.80 → 0.70   (약한 마스크도 살림)
  - stability_score_thresh: 0.80 → 0.75 (경계 모호한 새싹 살림)
  - min_mask_region_area: 5 → 2px  (1-2cm² 새싹 살림)

흐름은 v1과 동일. 결과는 _v4 접미사로 저장.
주의: torchvision NMS 디바이스 버그 위험 (0.23.0에서 fix 가능성 ↑).
       에러 시 즉시 fallback 권장.

실행: python -u scripts/sam_roi_test_v2.py
"""
from __future__ import annotations
from pathlib import Path
import time
import numpy as np
import rasterio
from rasterio.windows import Window
import cv2
import torch
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


ROOT = Path(r"C:/Users/user/Desktop/분석프로젝트/Soybean_Emergence_Rate")
FIELDS_DIR = ROOT / "result" / "fields"
SHP_DIR = ROOT / "shapefile"
OUT_DIR = ROOT / "result" / "sam_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = ROOT / "models"

# === ROI ===
ROI_SIZE_M = 30.0
DS_FACTOR = 1

# === 콩잎 필터 (v1 동일 — recall은 SAM 단에서 늘림) ===
MIN_LEAF_AREA_CM2 = 1.0
MAX_LEAF_AREA_CM2 = 200.0
HUE_MIN, HUE_MAX = 20, 100
SAT_MIN = 20
VAL_MIN = 30

# === SAM 강화 파라미터 (v2 핵심) ===
PTS_PER_SIDE = 64                  # 48 → 64
PRED_IOU_THRESH = 0.70             # 0.80 → 0.70
STABILITY_SCORE_THRESH = 0.75      # 0.80 → 0.75
MIN_MASK_REGION_AREA = 2           # 5 → 2

# 출력 버전 태그
VERSION_TAG = "v4"

SAM_CKPT = MODELS_DIR / "sam_vit_l_0b3195.pth"
SAM_TYPE = "vit_l"


def read_roi(tif_path: Path, roi_size_m: float):
    with rasterio.open(tif_path) as src:
        gsd_m = abs(src.transform.a)
        cy, cx = src.height // 2, src.width // 2
        roi_px = int(roi_size_m / gsd_m)
        y0 = max(0, cy - roi_px // 2); x0 = max(0, cx - roi_px // 2)
        y1 = min(src.height, y0 + roi_px); x1 = min(src.width, x0 + roi_px)
        window = Window(col_off=x0, row_off=y0, width=x1-x0, height=y1-y0)
        rgb = src.read(indexes=[1, 2, 3], window=window)
        alpha = src.read(4, window=window) if src.count >= 4 else None
        win_transform = src.window_transform(window)
    rgb = np.transpose(rgb, (1, 2, 0))
    if alpha is not None:
        rgb[alpha == 0] = 0
    return rgb, gsd_m, win_transform, (y0, x0)


def downsample(rgb: np.ndarray, ds: int):
    H, W = rgb.shape[:2]
    return cv2.resize(rgb, (W // ds, H // ds), interpolation=cv2.INTER_AREA)


def run_sam_on_image(image: np.ndarray, sam):
    """v2 강화 파라미터."""
    mask_gen = SamAutomaticMaskGenerator(
        sam,
        points_per_side=PTS_PER_SIDE,
        pred_iou_thresh=PRED_IOU_THRESH,
        stability_score_thresh=STABILITY_SCORE_THRESH,
        crop_n_layers=0,
        min_mask_region_area=MIN_MASK_REGION_AREA,
    )
    return mask_gen.generate(image)


def filter_soybean_leaves(masks: list, image_rgb: np.ndarray, gsd_m: float):
    px_area_cm2 = (gsd_m * 100) ** 2
    image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    accepted = []
    for m in masks:
        seg = m["segmentation"]
        area_px = int(seg.sum())
        area_cm2 = area_px * px_area_cm2
        if area_cm2 < MIN_LEAF_AREA_CM2 or area_cm2 > MAX_LEAF_AREA_CM2:
            continue
        h_mean = float(image_hsv[seg, 0].mean())
        s_mean = float(image_hsv[seg, 1].mean())
        v_mean = float(image_hsv[seg, 2].mean())
        if not (HUE_MIN <= h_mean <= HUE_MAX and s_mean >= SAT_MIN and v_mean >= VAL_MIN):
            continue
        accepted.append({**m, "area_cm2": area_cm2,
                         "h": h_mean, "s": s_mean, "v": v_mean})
    return accepted


def visualize(image_rgb: np.ndarray, leaves: list, all_masks: list,
              out_png: Path, title: str):
    fig, axes = plt.subplots(1, 3, figsize=(22, 8))
    DISP_MAX = 1500
    H, W = image_rgb.shape[:2]
    step = max(1, max(H, W) // DISP_MAX)
    rgb_ds = image_rgb[::step, ::step]
    Hd, Wd = rgb_ds.shape[:2]

    axes[0].imshow(rgb_ds)
    axes[0].set_title("(1) 원본 RGB ROI")
    axes[0].axis("off")

    axes[1].imshow(rgb_ds)
    overlay = np.zeros((Hd, Wd, 4), dtype=np.float32)
    rng = np.random.default_rng(0)
    for m in all_masks:
        seg_ds = m["segmentation"][::step, ::step]
        if seg_ds.sum() == 0:
            continue
        color = rng.random(3)
        overlay[seg_ds, :3] = color
        overlay[seg_ds, 3] = 0.45
    axes[1].imshow(overlay)
    axes[1].set_title(f"(2) SAM 전체 마스크 ({len(all_masks)}개)")
    axes[1].axis("off")

    axes[2].imshow(rgb_ds)
    xs_all, ys_all = [], []
    for L in leaves:
        ys, xs = np.where(L["segmentation"])
        if len(xs) == 0:
            continue
        xs_all.append(xs.mean() / step)
        ys_all.append(ys.mean() / step)
    axes[2].scatter(xs_all, ys_all, s=20, facecolors="none",
                    edgecolors="red", lw=1.2)
    axes[2].set_title(f"(3) 콩잎 필터 후 ({len(xs_all)}개)")
    axes[2].axis("off")
    axes[2].text(0.01, 0.99, f"total = {len(xs_all)}",
                 transform=axes[2].transAxes,
                 fontsize=14, fontweight="bold",
                 color="white", verticalalignment="top",
                 bbox=dict(boxstyle="round,pad=0.3",
                           facecolor="black", alpha=0.6))

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close()


def main():
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    print("=" * 60)
    print(f"SAM ROI v2 (강화) — GJSM-1-1 중심부  → {VERSION_TAG}")
    print(f"  points={PTS_PER_SIDE}, iou={PRED_IOU_THRESH}, "
          f"stab={STABILITY_SCORE_THRESH}, min_area={MIN_MASK_REGION_AREA}px")
    print("=" * 60)

    tif_path = FIELDS_DIR / "GJSM-1-1.tif"
    t0 = time.time()
    rgb, gsd_orig, win_transform, (y0, x0) = read_roi(tif_path, ROI_SIZE_M)
    print(f"[ROI] {time.time()-t0:.1f}s  shape={rgb.shape}, GSD={gsd_orig*1000:.2f}mm/px")

    t0 = time.time()
    rgb_ds = downsample(rgb, DS_FACTOR)
    gsd_ds = gsd_orig * DS_FACTOR
    print(f"[ds]  {time.time()-t0:.1f}s  shape={rgb_ds.shape}, GSD={gsd_ds*1000:.2f}mm/px")

    t0 = time.time()
    sam = sam_model_registry[SAM_TYPE](checkpoint=str(SAM_CKPT))
    sam.to("cuda")
    print(f"[SAM] 모델 로드 {time.time()-t0:.1f}s")

    H, W = rgb_ds.shape[:2]
    print(f"[run] 영상 {H}×{W}")
    if max(H, W) <= 1024:
        t0 = time.time()
        all_masks = run_sam_on_image(rgb_ds, sam)
        print(f"[run] SAM (single) {time.time()-t0:.1f}s  마스크 {len(all_masks)}개")
    else:
        tile = 1024; stride = tile - 128
        all_masks = []
        n_tiles = ((H - 1) // stride + 1) * ((W - 1) // stride + 1)
        print(f"[run] 타일 분할 ({tile}px, stride {stride}) 약 {n_tiles}개")
        t0 = time.time()
        for ty in range(0, H, stride):
            for tx in range(0, W, stride):
                y1 = min(ty + tile, H); x1 = min(tx + tile, W)
                if y1 - ty < 100 or x1 - tx < 100:
                    continue
                tile_rgb = rgb_ds[ty:y1, tx:x1].copy()
                try:
                    tile_masks = run_sam_on_image(tile_rgb, sam)
                except RuntimeError as e:
                    msg = str(e)
                    if "indices should be either on cpu" in msg or "device" in msg:
                        print(f"  ❌ NMS device bug 발생 (tile y={ty},x={tx}): {msg[:80]}")
                        print(f"     → fallback 필요. 스크립트 종료.")
                        return
                    raise
                for m in tile_masks:
                    seg_full = np.zeros((H, W), dtype=bool)
                    seg_full[ty:y1, tx:x1] = m["segmentation"]
                    m["segmentation"] = seg_full
                    all_masks.append(m)
                print(f"  tile (y={ty},x={tx}) → {len(tile_masks)}개, 누적 {len(all_masks)}")
        print(f"[run] SAM (tiled) {time.time()-t0:.1f}s  총 마스크 {len(all_masks)}")

    t0 = time.time()
    leaves = filter_soybean_leaves(all_masks, rgb_ds, gsd_ds)
    print(f"[filter] {time.time()-t0:.1f}s  콩잎 {len(leaves)}개 (전체 {len(all_masks)} 중)")

    t1 = time.time()
    leaf_data = []
    for L in leaves:
        ys, xs = np.where(L["segmentation"])
        if len(xs) == 0:
            continue
        leaf_data.append([
            ys.mean(), xs.mean(),
            L["area_cm2"],
            L.get("h", 0), L.get("s", 0), L.get("v", 0),
            ys.min(), xs.min(), ys.max(), xs.max(),
        ])
    leaf_arr = np.array(leaf_data, dtype=np.float32)
    DISP_MAX = 1500
    H_im, W_im = rgb_ds.shape[:2]
    step_im = max(1, max(H_im, W_im) // DISP_MAX)
    rgb_disp = rgb_ds[::step_im, ::step_im]

    npz_path = OUT_DIR / f"GJSM-1-1_sam_roi_test_ds{DS_FACTOR}_{VERSION_TAG}.npz"
    np.savez_compressed(
        npz_path,
        rgb_disp=rgb_disp,
        rgb_step=np.array([step_im]),
        leaf_arr=leaf_arr,
        gsd_ds=np.array([gsd_ds]),
        n_all_masks=np.array([len(all_masks)]),
    )
    print(f"[npz]  {time.time()-t1:.1f}s  {npz_path.name} ({npz_path.stat().st_size/1e6:.1f}MB)")

    out_png = OUT_DIR / f"GJSM-1-1_sam_roi_test_ds{DS_FACTOR}_{VERSION_TAG}.png"
    visualize(rgb_ds, leaves, all_masks, out_png,
              title=f"SAM ROI {VERSION_TAG} (강화) — GJSM-1-1 {ROI_SIZE_M}m²  "
                    f"pts={PTS_PER_SIDE}, iou={PRED_IOU_THRESH}, stab={STABILITY_SCORE_THRESH}")
    print(f"[save] {out_png}")

    if leaves:
        area_arr = np.array([L["area_cm2"] for L in leaves])
        print(f"\n=== 콩잎 통계 ({VERSION_TAG}) ===")
        print(f"  개수: {len(leaves)}")
        print(f"  면적 (cm²): mean={area_arr.mean():.2f}, "
              f"min={area_arr.min():.2f}, max={area_arr.max():.2f}")
        roi_area_m2 = ROI_SIZE_M * ROI_SIZE_M
        density_per_ha = len(leaves) / (roi_area_m2 / 10000)
        print(f"  밀도: {density_per_ha:,.0f} 개체/ha")
        print(f"  표준(65cm×20cm) 100% 입모: 76,923 개체/ha")
        print(f"  추정 입모율: {density_per_ha / 76923 * 100:.1f}%")
        print(f"\n  ※ v1(_v3) 기준: 4694개 → 67.8% 였음")


if __name__ == "__main__":
    main()
