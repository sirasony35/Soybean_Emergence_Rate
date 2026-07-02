"""
콩 입모율 파이프라인 공통 유틸.

좌표계 규약:
  - 원본 픽셀 좌표 (y, x): SAM leaf_arr 그대로. 원본 TIF와 동일 스케일.
  - disp 좌표: rgb_disp 이미지 좌표. rgb_step배 다운샘플 = 5.25cm/px 스케일.
  - 실측 (m, cm): gsd_m × 픽셀 = 미터. px_to_cm = gsd_m × 100.

두둑 좌표계:
  - ridge_dir: 두둑이 뻗은 방향 (단위벡터, y·x)
  - perp_dir: 두둑에 수직인 방향 (단위벡터, y·x)
  - a_deg: 스캔 각도 (0~180°). perp_dir = (cos a, sin a), ridge_dir = (-sin a, cos a)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


ROOT = Path(r"C:/Users/user/Desktop/분석프로젝트/Soybean_Emergence_Rate")
FIELDS_DIR = ROOT / "result" / "fields"           # SAM 입력 필지 TIF
NPZ_DIR = ROOT / "result" / "sam_leaves"          # SAM 콩잎 검출 npz
OUT_DIR = ROOT / "result" / "pipeline"            # 파이프라인 산출
REPORT_DIR = ROOT / "result" / "report"           # 파종기 비교 리포트
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def setup_korean_font():
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False


def load_field_npz(field: str) -> dict:
    """SAM v4 검출 결과 로드. FULL 우선, ROI fallback."""
    candidates = [
        NPZ_DIR / f"{field}_sam_FULL_v4.npz",
        NPZ_DIR / f"{field}_sam_roi_test_ds1_v4.npz",
    ]
    npz_path = next((c for c in candidates if c.exists()), None)
    if npz_path is None:
        raise FileNotFoundError(f"npz 없음: {field}")
    d = np.load(npz_path)
    return dict(
        npz_path=npz_path,
        rgb_disp=d["rgb_disp"],
        rgb_step=int(d["rgb_step"][0]),
        leaves=d["leaf_arr"],
        gsd_m=float(d["gsd_ds"][0]),
        valid_area_ha=float(d["valid_area_ha"][0]) if "valid_area_ha" in d.files else None,
        field_area_ha=float(d["field_area_ha"][0]) if "field_area_ha" in d.files else None,
        n_all_masks=int(d["n_all_masks"][0]),
    )


def leaf_centroids(leaves: np.ndarray) -> np.ndarray:
    """leaf_arr → (N, 2) [y, x] 원본 픽셀 좌표."""
    return leaves[:, :2]


def get_field_dir(field: str) -> Path:
    """필지별 파이프라인 출력 폴더."""
    p = OUT_DIR / field
    p.mkdir(parents=True, exist_ok=True)
    return p


def angle_perp_ridge(a_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """
    스캔 각도 a_deg (0~180°) 에서 perp/ridge 단위벡터.
      perp_dir = (cos a, sin a)     — 두둑에 수직 (density profile 축)
      ridge_dir = (-sin a, cos a)   — 두둑이 뻗은 방향
    a_deg=0  → 두둑 수평 (perp=(1,0)=y축 방향)
    a_deg=90 → 두둑 수직 (perp=(0,1)=x축 방향)
    """
    a = np.radians(a_deg)
    perp = np.array([np.cos(a), np.sin(a)])
    ridge = np.array([-np.sin(a), np.cos(a)])
    return perp, ridge


def project(points: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """points (N, 2) 을 unit vector direction 위로 스칼라 사영."""
    return points @ direction
