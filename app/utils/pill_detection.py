# pill_detection.py  — 無 rembg 版本（YOLO → 顏色/外型 → OCR）

import os
import base64
import logging
import cv2
import torch
from ultralytics import YOLO

from app.utils.image_io import read_image_safely
from openocr import OpenOCR
from app.utils.ocr_utils import recognize_with_openocr
from app.utils.shape_color_utils import (
    rotate_image_by_angle,
    enhance_contrast,
    desaturate_image,
    enhance_for_blur,
    # extract_dominant_colors_by_ratio,
    get_basic_color_name,
    get_dominant_colors
    detect_shape_from_image
)
from app.utils.logging_utils import log_mem

# ====== 輕量化設定 ======
# Render 的 CPU 只有 1 核，避免 PyTorch/NumPy 開太多執行緒
torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "1")))

logging.getLogger("openrec").setLevel(logging.ERROR)
ocr_engine = OpenOCR(backend='onnx', device='cpu')

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
_det_model = None


def get_det_model():
    """Lazy-load YOLO 權重，只初始化一次"""
    global _det_model
    if _det_model is None:
        print("[DET] loading YOLO model…")
        m = YOLO("models/best.pt")
        try:
            m.fuse()
        except Exception:
            pass
        _det_model = m
        print("[DET] model ready")
    return _det_model


def generate_image_versions(base_img):
    """產生多個影像增強版本供 OCR 嘗試"""
    v1 = enhance_contrast(base_img, 1.5, 1.5, -0.5)
    # 減少判斷
    # v2 = desaturate_image(v1)
    # v3 = enhance_contrast(base_img, 5.5, 2.0, -1.0)
    # v4 = desaturate_image(v3)
    # v5 = enhance_for_blur(base_img)
    # 減少判斷
    # return [
    #     (base_img, "原圖"),
    #     (v1, "增強1"),
    #     (v2, "去飽和1"),
    #     (v3, "增強2"),
    #     (v4, "去飽和2"),
    #     (v5, "模糊優化"),
    # ]
    return [
        (base_img, "原圖"),
        (v1, "增強去飽和"),
    ]


def get_best_ocr_texts(
        image_versions,
        # angles=(0, 45, 90, 135, 180, 225, 270, 315), ocr_engine=None,
        angles=(0, 90, 180, 270), ocr_engine=None,
):
    version_results = {}
    score_dict = {}

    for img_v, version_name in image_versions:
        for angle in angles:
            rotated = rotate_image_by_angle(img_v, angle)
            full_name = f"{version_name}_旋轉{angle}"
            texts, score = recognize_with_openocr(
                rotated, ocr_engine=ocr_engine, name=full_name, min_score=0.8
            )
            version_results[full_name] = texts
            score_dict[full_name] = score

    score_combined = {
        k: (sum(len(txt) for txt in version_results[k]) * score_dict[k])
        for k in version_results
    }
    best_name = max(score_combined, key=score_combined.get)
    return version_results[best_name], best_name, score_dict[best_name]


def _pick_crop_from_boxes(input_img, boxes):
    """從 YOLO boxes 選最佳框並回傳裁切圖（不再去背）"""
    xyxy = boxes.xyxy.cpu().numpy()  # [N,4]
    conf = boxes.conf.squeeze().cpu().numpy()
    conf = conf if conf.ndim else conf[None]

    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    score = conf * (areas / (areas.max() + 1e-6))  # 面積加權，避免挑到超小框
    best_idx = score.argmax()
    x1, y1, x2, y2 = map(int, xyxy[best_idx])

    pad = int(0.08 * max(x2 - x1, y2 - y1))  # 少量 padding
    h, w = input_img.shape[:2]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w - 1, x2 + pad)
    y2 = min(h - 1, y2 + pad)

    cropped = input_img[y1:y2, x1:x2]
    return cropped


def process_image(img_path: str):
    """
    單張藥品圖片辨識流程：
    YOLO → 裁切 → 顏色/外型 → 多版本 OCR → 回傳
    """
    print(f"[PROC] start process_image: {img_path}")
    log_mem("infer:start")
    det_model = get_det_model()
    log_mem("infer:after_get_model")
    # === 讀取圖片 ===
    input_img = read_image_safely(img_path)
    if input_img is None:
        return {"error": "無法讀取圖片"}

    # === YOLO 偵測（先正常閾值，失敗再降閾值）===
    print("[PROC] YOLO predict (conf=0.25)…")
    res = det_model.predict(
        source=input_img, imgsz=640, conf=0.25, iou=0.7, device=DEVICE, verbose=False
    )[0]
    boxes = res.boxes

    if boxes is not None and boxes.xyxy.shape[0] > 0:
        cropped = _pick_crop_from_boxes(input_img, boxes)
    else:
        print("[PROC] no box, try lower conf=0.10…")
        res_lo = det_model.predict(
            source=input_img, imgsz=640, conf=0.10, iou=0.7, device=DEVICE, verbose=False
        )[0]
        log_mem("infer:after_predict")
        boxes_lo = res_lo.boxes
        if boxes_lo is not None and boxes_lo.xyxy.shape[0] > 0:
            cropped = _pick_crop_from_boxes(input_img, boxes_lo)
        else:
            print("[PROC] detection failed — return early (no rembg fallback).")
            return {"error": "藥品擷取失敗"}

    # === 裁切圖轉 Base64（給前端顯示） ===
    ok, buffer = cv2.imencode(".jpg", cropped)
    cropped_b64 = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}" if ok else None

    # === 外型、顏色分析（直接用裁切圖，不去背） ===
    shape, _ = detect_shape_from_image(cropped, cropped, expected_shape=None)

    rgb_colors, hex_colors = get_dominant_colors(img, k=3, min_ratio=0.30)
    rgb_colors_int = [tuple(map(int, c)) for c in rgb_colors]
    basic_names = []
    hsv_values = []
    for rgb in rgb_colors_int:
        bgr = np.uint8([[rgb[::-1]]])
        h_raw, s, v = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0][0]
        h_deg = h_raw * 2
        hsv_values.append((h_deg, s, v))
        
        cname = get_basic_color_name(rgb)
        basic_names.append(cname)
        
    colors = list(dict.fromkeys(basic_names))

    # === 多版本 OCR 辨識 ===
    image_versions = generate_image_versions(cropped)
    best_texts, best_name, best_score = get_best_ocr_texts(
        image_versions, ocr_engine=ocr_engine
    )

    print(f"[PROC] OCR={best_texts}, shape={shape}, colors={colors}, score={best_score:.3f}")

    return {
        "文字辨識": best_texts if best_texts else ["None"],
        "最佳版本": best_name,
        "信心分數": round(best_score, 3),
        "顏色": colors,
        "外型": shape,
        "cropped_image": cropped_b64,
    }
