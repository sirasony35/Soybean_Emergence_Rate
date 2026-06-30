"""
SAM ROI 테스트 — GJSM-1-1 중심부 30m × 30m

흐름:
  1) GJSM-1-1.tif에서 필지 중심부 30m × 30m ROI 추출
  2) 2배 다운샘플(5.25mm → 10.5mm = 1cm GSD)
  3) 1024×1024 타일 단위로 SAM Automatic Mask Generator 적용
  4) 콩잎 필터링:
       - 면적 4-100cm² (어린 새싹부터 자란 잎)
       - 녹색 색조 (HSV Hue 30-90, S>30, V>40)
  5) 시각화: RGB 위 빨간점(콩잎 중심) + 카운트
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

# === ROI 파라미터 ===
ROI_SIZE_M = 30.0       # 30m × 30m
DS_FACTOR = 1           # 원본 해상도 (5.25mm GSD) — 콩 새싹 4-6 픽셀

# === 콩잎 필터 (완화) ===
MIN_LEAF_AREA_CM2 = 1.0     # 어린 새싹 (1cm² 이상)
MAX_LEAF_AREA_CM2 = 200.0   # 큰 잎까지 (14cm × 14cm)
HUE_MIN, HUE_MAX = 20, 100  # HSV 녹색 범위 확장
SAT_MIN = 20
VAL_MIN = 30

# SAM 모델
SAM_CKPT = MODELS_DIR / "sam_vit_l_0b3195.pth"
SAM_TYPE = "vit_l"


def read_roi(tif_path: Path, roi_size_m: float):
    """필지 중심부 roi_size_m × roi_size_m ROI 추출 → RGB array."""
    with rasterio.open(tif_path) as src:
        gsd_m = abs(src.transform.a)
        cy, cx = src.height // 2, src.width // 2
        roi_px = int(roi_size_m / gsd_m)
        y0 = max(0, cy - roi_px // 2); x0 = max(0, cx - roi_px // 2)
        y1 = min(src.height, y0 + roi_px); x1 = min(src.width, x0 + roi_px)
        window = Window(col_off=x0, row_off=y0, width=x1-x0, height=y1-y0)
        rgb = src.read(indexes=[1,2,3], window=window)  # (3, H, W) uint8
        alpha = src.read(4, window=window) if src.count >= 4 else None
        win_transform = src.window_transform(window)
    rgb = np.transpose(rgb, (1, 2, 0))  # (H, W, 3)
    if alpha is not None:
        rgb[alpha == 0] = 0
    return rgb, gsd_m, win_transform, (y0, x0)


def downsample(rgb: np.ndarray, ds: int):
    """블록 평균 다운샘플 (cv2.resize INTER_AREA)."""
    H, W = rgb.shape[:2]
    return cv2.resize(rgb, (W // ds, H // ds), interpolation=cv2.INTER_AREA)


def run_sam_on_image(image: np.ndarray, sam, points_per_side: int = 48,
                      min_area_px: int = 5):
    """SAM Automatic Mask Generator (검증 작동 설정 + 약간 완화)."""
    mask_gen = SamAutomaticMaskGenerator(
        sam,
        points_per_side=points_per_side,
        pred_iou_thresh=0.80,                 # 0.85 → 0.80 (살짝 완화)
        stability_score_thresh=0.80,          # 0.85 → 0.80
        crop_n_layers=0,
        min_mask_region_area=min_area_px,
    )
    return mask_gen.generate(image)


def filter_soybean_leaves(masks: list, image_rgb: np.ndarray, gsd_m: float,
                          min_area_cm2: float = MIN_LEAF_AREA_CM2,
                          max_area_cm2: float = MAX_LEAF_AREA_CM2):
    """면적·색조 기준으로 콩잎만 필터링."""
    px_area_cm2 = (gsd_m * 100) ** 2   # 한 픽셀 면적 (cm²)
    image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    accepted = []
    for m in masks:
        seg = m["segmentation"]  # bool array
        area_px = int(seg.sum())
        area_cm2 = area_px * px_area_cm2
        if area_cm2 < min_area_cm2 or area_cm2 > max_area_cm2:
            continue
        # 색조 평가 — 마스크 내 픽셀의 HSV 평균
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
    """시각화 최적화 — 다운샘플 색칠 + 일괄 scatter."""
    fig, axes = plt.subplots(1, 3, figsize=(22, 8))

    # 시각화는 다운샘플 (큰 영상이면)
    DISP_MAX = 1500
    H, W = image_rgb.shape[:2]
    step = max(1, max(H, W) // DISP_MAX)
    rgb_ds = image_rgb[::step, ::step]
    Hd, Wd = rgb_ds.shape[:2]

    # (1) 원본 RGB
    axes[0].imshow(rgb_ds)
    axes[0].set_title("(1) 원본 RGB ROI")
    axes[0].axis("off")

    # (2) SAM 전체 마스크 (다운샘플 + 랜덤 색 vectorize)
    axes[1].imshow(rgb_ds)
    overlay = np.zeros((Hd, Wd, 4), dtype=np.float32)
    rng = np.random.default_rng(0)
    for m in all_masks:
        seg_ds = m["segmentation"][::step, ::step]  # 다운샘플 마스크
        if seg_ds.sum() == 0:
            continue
        color = rng.random(3)
        overlay[seg_ds, :3] = color
        overlay[seg_ds, 3] = 0.45
    axes[1].imshow(overlay)
    axes[1].set_title(f"(2) SAM 전체 마스크 ({len(all_masks)}개)")
    axes[1].axis("off")

    # (3) 콩잎 빨간 점 — 미리 좌표 모아 한 번에 scatter
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
    # 좌상단 텍스트 (농진청 스타일)
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
    print("SAM ROI 테스트 — GJSM-1-1 중심부")
    print("=" * 60)

    # 1) ROI 추출
    tif_path = FIELDS_DIR / "GJSM-1-1.tif"
    t0 = time.time()
    rgb, gsd_orig, win_transform, (y0, x0) = read_roi(tif_path, ROI_SIZE_M)
    print(f"[ROI] {time.time()-t0:.1f}s  shape={rgb.shape}, GSD={gsd_orig*1000:.2f}mm/px")

    # 2) 다운샘플
    t0 = time.time()
    rgb_ds = downsample(rgb, DS_FACTOR)
    gsd_ds = gsd_orig * DS_FACTOR
    print(f"[ds]  {time.time()-t0:.1f}s  shape={rgb_ds.shape}, GSD={gsd_ds*1000:.2f}mm/px")

    # 3) SAM 모델 로드
    t0 = time.time()
    sam = sam_model_registry[SAM_TYPE](checkpoint=str(SAM_CKPT))
    sam.to("cuda")
    print(f"[SAM] 모델 로드 {time.time()-t0:.1f}s")

    # 4) SAM 적용 (전체 ROI에 한 번에 — 1024 이내라 충분)
    # 1024 초과면 타일 분할 필요. 30m / 0.0105m = 2857px → 타일 필요
    H, W = rgb_ds.shape[:2]
    print(f"[run] 영상 {H}×{W}")
    if max(H, W) <= 1024:
        t0 = time.time()
        all_masks = run_sam_on_image(rgb_ds, sam, points_per_side=48)
        print(f"[run] SAM (single) {time.time()-t0:.1f}s  마스크 {len(all_masks)}개")
    else:
        # 타일 분할 (1024 - 128 overlap 단위)
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
                tile_masks = run_sam_on_image(tile_rgb, sam, points_per_side=48)
                for m in tile_masks:
                    # 전체 좌표로 변환
                    seg_full = np.zeros((H, W), dtype=bool)
                    seg_full[ty:y1, tx:x1] = m["segmentation"]
                    m["segmentation"] = seg_full
                    all_masks.append(m)
                print(f"  tile (y={ty},x={tx}) → {len(tile_masks)}개, 누적 {len(all_masks)}")
        print(f"[run] SAM (tiled) {time.time()-t0:.1f}s  총 마스크 {len(all_masks)}")

    # 5) 콩잎 필터
    t0 = time.time()
    leaves = filter_soybean_leaves(all_masks, rgb_ds, gsd_ds)
    print(f"[filter] {time.time()-t0:.1f}s  콩잎 {len(leaves)}개 (전체 {len(all_masks)} 중)")

    # 6) 결과 npz 저장 (시각화 실패해도 재시도 가능)
    # 마스크 전체는 너무 큼 — centroid + area + bbox만 저장
    t1 = time.time()
    leaf_data = []
    for L in leaves:
        ys, xs = np.where(L["segmentation"])
        if len(xs) == 0: continue
        leaf_data.append([
            ys.mean(), xs.mean(),         # y, x centroid
            L["area_cm2"],
            L.get("h", 0), L.get("s", 0), L.get("v", 0),
            ys.min(), xs.min(), ys.max(), xs.max(),  # bbox
        ])
    leaf_arr = np.array(leaf_data, dtype=np.float32)
    # 다운샘플된 RGB만 저장 (시각화 표시용)
    DISP_MAX = 1500
    H_im, W_im = rgb_ds.shape[:2]
    step_im = max(1, max(H_im, W_im) // DISP_MAX)
    rgb_disp = rgb_ds[::step_im, ::step_im]

    npz_path = OUT_DIR / f"GJSM-1-1_sam_roi_test_ds{DS_FACTOR}_v3.npz"
    np.savez_compressed(
        npz_path,
        rgb_disp=rgb_disp,             # 표시용 다운샘플 RGB
        rgb_step=np.array([step_im]),  # 원본→표시 다운샘플 비율
        leaf_arr=leaf_arr,             # 콩잎 (y, x, area, h, s, v, y0, x0, y1, x1)
        gsd_ds=np.array([gsd_ds]),
        n_all_masks=np.array([len(all_masks)]),
    )
    print(f"[npz]  {time.time()-t1:.1f}s  {npz_path.name} ({npz_path.stat().st_size/1e6:.1f}MB)")

    # 7) 시각화
    out_png = OUT_DIR / f"GJSM-1-1_sam_roi_test_ds{DS_FACTOR}_v3.png"
    visualize(rgb_ds, leaves, all_masks, out_png,
              title=f"SAM ROI 테스트 — GJSM-1-1 중심부 {ROI_SIZE_M}m×{ROI_SIZE_M}m "
                    f"(GSD={gsd_ds*1000:.1f}mm/px)")
    print(f"[save] {out_png}")

    # 7) 요약
    if leaves:
        area_arr = np.array([L["area_cm2"] for L in leaves])
        print(f"\n=== 콩잎 통계 ===")
        print(f"  개수: {len(leaves)}")
        print(f"  면적 (cm²): mean={area_arr.mean():.2f}, "
              f"min={area_arr.min():.2f}, max={area_arr.max():.2f}")
        roi_area_m2 = ROI_SIZE_M * ROI_SIZE_M
        density_per_ha = len(leaves) / (roi_area_m2 / 10000)
        print(f"  밀도: {density_per_ha:,.0f} 개체/ha")
        # 표준 비교: 65cm × 20cm 파종 = 76,923 개체/ha
        print(f"  표준(65cm×20cm) 100% 입모: 76,923 개체/ha")
        print(f"  추정 입모율: {density_per_ha / 76923 * 100:.1f}%")


if __name__ == "__main__":
    main()
