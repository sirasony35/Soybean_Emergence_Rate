"""
SAM v2 필지 처리 — 단일 필지 30m ROI에서 강화 파라미터로 SAM 실행 + 기본 필터.
성능 최적화: 타일 레벨에서 centroid/bbox 계산 → 전역 mask 저장 X → npz 저장 초고속

사용:
  python -u scripts/sam_run_field.py <field_name>
  python -u scripts/sam_run_field.py GJSM-1-1_Smart
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window
import cv2
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


ROOT = Path(r"C:/Users/user/Desktop/분석프로젝트/Soybean_Emergence_Rate")
FIELDS_DIR = ROOT / "result" / "fields"
OUT_DIR = ROOT / "result" / "sam_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = ROOT / "models"

# ROI
ROI_SIZE_M = 30.0
DS_FACTOR = 1

# 콩잎 기본 필터 (v1 원본)
MIN_LEAF_AREA_CM2 = 1.0
MAX_LEAF_AREA_CM2 = 200.0
HUE_MIN, HUE_MAX = 20, 100
SAT_MIN = 20
VAL_MIN = 30

# SAM v2 강화 파라미터
PTS_PER_SIDE = 64
PRED_IOU_THRESH = 0.70
STABILITY_SCORE_THRESH = 0.75
MIN_MASK_REGION_AREA = 2

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
    rgb = np.transpose(rgb, (1, 2, 0))
    if alpha is not None:
        rgb[alpha == 0] = 0
    return rgb, gsd_m


def make_mask_gen(sam):
    return SamAutomaticMaskGenerator(
        sam,
        points_per_side=PTS_PER_SIDE,
        pred_iou_thresh=PRED_IOU_THRESH,
        stability_score_thresh=STABILITY_SCORE_THRESH,
        crop_n_layers=0,
        min_mask_region_area=MIN_MASK_REGION_AREA,
    )


def process_tile(tile_rgb: np.ndarray, tile_hsv: np.ndarray, mask_gen,
                  ty: int, tx: int, gsd_m: float):
    """
    타일에 SAM 적용 + 콩잎 필터 (area + HSV) → 통과한 잎만 반환.
    잎당 metadata: [y_cent, x_cent, area_cm2, h, s, v, y0, x0, y1, x1] (전역 좌표)
    """
    px_area_cm2 = (gsd_m * 100) ** 2
    tile_masks = mask_gen.generate(tile_rgb)
    n_total = len(tile_masks)
    leaves = []
    for m in tile_masks:
        seg = m["segmentation"]                      # (th, tw) bool
        area_px = int(m["area"]) if "area" in m else int(seg.sum())
        area_cm2 = area_px * px_area_cm2
        if area_cm2 < MIN_LEAF_AREA_CM2 or area_cm2 > MAX_LEAF_AREA_CM2:
            continue
        h_mean = float(tile_hsv[seg, 0].mean())
        s_mean = float(tile_hsv[seg, 1].mean())
        v_mean = float(tile_hsv[seg, 2].mean())
        if not (HUE_MIN <= h_mean <= HUE_MAX and s_mean >= SAT_MIN and v_mean >= VAL_MIN):
            continue
        # 전역 centroid, bbox — 타일 좌표계에서 계산 후 오프셋
        ys, xs = np.where(seg)
        leaves.append([
            float(ys.mean()) + ty, float(xs.mean()) + tx,
            area_cm2,
            h_mean, s_mean, v_mean,
            int(ys.min()) + ty, int(xs.min()) + tx,
            int(ys.max()) + ty, int(xs.max()) + tx,
        ])
    return leaves, n_total


def main():
    if len(sys.argv) < 2:
        print("사용: python sam_run_field.py <field_name>")
        print("예:   python sam_run_field.py GJSM-1-1_Smart")
        sys.exit(1)
    field_name = sys.argv[1]
    tif_path = FIELDS_DIR / f"{field_name}.tif"
    if not tif_path.exists():
        print(f"❌ TIF 없음: {tif_path}")
        sys.exit(1)

    print("=" * 60)
    print(f"SAM v2 필지 처리 — {field_name}  → {VERSION_TAG}")
    print(f"  pts={PTS_PER_SIDE}, iou={PRED_IOU_THRESH}, "
          f"stab={STABILITY_SCORE_THRESH}, min_area={MIN_MASK_REGION_AREA}px")
    print("=" * 60)

    t0 = time.time()
    rgb, gsd_orig = read_roi(tif_path, ROI_SIZE_M)
    print(f"[ROI] {time.time()-t0:.1f}s  shape={rgb.shape}, GSD={gsd_orig*1000:.2f}mm/px")

    gsd_ds = gsd_orig * DS_FACTOR
    rgb_ds = rgb  # DS_FACTOR=1이므로 그대로

    t0 = time.time()
    sam = sam_model_registry[SAM_TYPE](checkpoint=str(SAM_CKPT))
    sam.to("cuda")
    mask_gen = make_mask_gen(sam)
    print(f"[SAM] 모델 로드 {time.time()-t0:.1f}s")

    H, W = rgb_ds.shape[:2]
    print(f"[run] 영상 {H}×{W}")
    all_leaves = []
    total_masks = 0

    if max(H, W) <= 1024:
        tile_hsv = cv2.cvtColor(rgb_ds, cv2.COLOR_RGB2HSV)
        t0 = time.time()
        leaves, n_total = process_tile(rgb_ds, tile_hsv, mask_gen, 0, 0, gsd_ds)
        all_leaves.extend(leaves)
        total_masks += n_total
        print(f"[run] SAM {time.time()-t0:.1f}s  마스크 {n_total}, 필터 통과 {len(leaves)}")
    else:
        tile = 1024
        stride = tile - 128
        n_tiles = ((H - 1) // stride + 1) * ((W - 1) // stride + 1)
        print(f"[run] 타일 분할 ({tile}px, stride {stride}) 약 {n_tiles}개")
        t0 = time.time()
        tile_idx = 0
        for ty in range(0, H, stride):
            for tx in range(0, W, stride):
                y1 = min(ty + tile, H); x1 = min(tx + tile, W)
                if y1 - ty < 100 or x1 - tx < 100:
                    continue
                tile_idx += 1
                tile_rgb = rgb_ds[ty:y1, tx:x1].copy()
                tile_hsv = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2HSV)
                try:
                    leaves, n_total = process_tile(tile_rgb, tile_hsv, mask_gen,
                                                    ty, tx, gsd_ds)
                except RuntimeError as e:
                    msg = str(e)
                    if "indices should be either on cpu" in msg or "device" in msg:
                        print(f"  ❌ NMS device bug (tile y={ty},x={tx}): {msg[:80]}")
                        return
                    raise
                all_leaves.extend(leaves)
                total_masks += n_total
                print(f"  tile {tile_idx}/{n_tiles} (y={ty},x={tx}) → "
                      f"마스크 {n_total}, 잎 {len(leaves)}, 누적 잎 {len(all_leaves)}")
        print(f"[run] SAM (tiled) {time.time()-t0:.1f}s  총 마스크 {total_masks}, "
              f"총 잎 {len(all_leaves)}")

    # ─── npz 저장 (초고속: 이미 전부 계산됨) ───
    t0 = time.time()
    leaf_arr = np.array(all_leaves, dtype=np.float32) if all_leaves \
                else np.zeros((0, 10), dtype=np.float32)
    DISP_MAX = 1500
    step_im = max(1, max(H, W) // DISP_MAX)
    rgb_disp = rgb_ds[::step_im, ::step_im]

    npz_path = OUT_DIR / f"{field_name}_sam_roi_test_ds{DS_FACTOR}_{VERSION_TAG}.npz"
    np.savez_compressed(
        npz_path,
        rgb_disp=rgb_disp,
        rgb_step=np.array([step_im]),
        leaf_arr=leaf_arr,
        gsd_ds=np.array([gsd_ds]),
        n_all_masks=np.array([total_masks]),
    )
    print(f"[npz]  {time.time()-t0:.1f}s  {npz_path.name} ({npz_path.stat().st_size/1e6:.1f}MB)")

    n_leaves = len(all_leaves)
    if n_leaves > 0:
        roi_area_m2 = ROI_SIZE_M * ROI_SIZE_M
        density_per_ha = n_leaves / (roi_area_m2 / 10000)
        print(f"\n=== {field_name} 결과 ===")
        print(f"  잎 개수: {n_leaves}")
        print(f"  밀도: {density_per_ha:,.0f} 개체/ha")
        print(f"  추정 입모율 (baseline): {density_per_ha / 76923 * 100:.1f}%")


if __name__ == "__main__":
    main()
