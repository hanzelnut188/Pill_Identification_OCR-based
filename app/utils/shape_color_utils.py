import cv2

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import Counter
def rotate_image_by_angle(image, angle):
    """
    將圖片依指定角度旋轉。
    - image: 輸入圖片（OpenCV 格式）
    - angle: 順時針旋轉角度（例如 90, 180）
    - return: 旋轉後的圖片
    """
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return rotated


def enhance_contrast(img, clip_limit, alpha, beta):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    enhance_img = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    blurred = cv2.GaussianBlur(enhance_img, (5, 5), 3.0)
    return cv2.addWeighted(enhance_img, alpha, blurred, beta, 0)

def rgb_to_hex(color):
    """Convert RGB (0-255) to HEX string."""
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

def get_center_region(img, size=100):
    """
    擷取圖片的中央區域 (固定大小)。
    - img: 輸入圖片 (H, W, C)
    - size: 方形區域邊長 (像素)，預設 100
    - return: 中央裁切後的圖片
    """
    h, w = img.shape[:2]

    cx, cy = w // 2, h // 2  # 圖片中心點

    # 計算邊界
    x1 = max(cx - size // 2, 0)
    y1 = max(cy - size // 2, 0)
    x2 = min(cx + size // 2, w)
    y2 = min(cy + size // 2, h)

    return img[y1:y2, x1:x2]

def is_color_similar(hsv1, hsv2, h_thresh=20, s_thresh=40, v_thresh=40):
    """
    Check if two HSV colors are similar within thresholds.
    Hue in degrees (0-360), s & v in [0-255].
    """
    h1, s1, v1 = hsv1
    h2, s2, v2 = hsv2

    # circular hue difference (because 0° ≈ 360°)
    dh = min(abs(h1 - h2), 360 - abs(h1 - h2))
    ds = abs(s1 - s2)
    dv = abs(v1 - v2)

    return (dh <= h_thresh) and (ds <= s_thresh) and (dv <= v_thresh)

def get_basic_color_name(rgb):
    """Classify an RGB value into a basic color family (Chinese Traditional)."""
    # Convert RGB (0–255) to HSV
    bgr = np.uint8([[rgb[::-1]]])  # OpenCV expects BGR
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = hsv
    print(f"HSV: {h}, {s}, {v}")
    h = int(h)*2   # Convert to degrees (0-360)
    r, g, b = rgb
    print(f"RGB: {r}, {g}, {b}")
    # Handle black/white/gray
    # if v < 60:   # instead of 40 → more tolerant to lighting
    #     return "黑色"
    if s < 40 and v > 170:
        return "白色"
    if s < 40:
        return "灰色"

    # Hue ranges
    if (h < 10 or h >= 330) and s > 90 and r > 50:
        return "紅色"
    elif h < 30:
        return "棕色" if v < 150 else "橙色"
    elif h < 60:
        return "黃色"
    elif h < 250 and g > b:
        return "綠色"
    elif h < 250 and g < b:
        return "藍色"
    elif h < 300:  # candidate for pink (270–330)
        return "紫色"
    elif h < 360:  # candidate for pink (270–330)
        return "粉紅色"            
    else:
        return "其他"


def get_image_rgb(path: str):
    """
    Load an image (jpg/png/heic) and return as RGB numpy array.
    """
    try:
        # Use PIL so HEIC is supported
        pil_img = Image.open(path).convert("RGB")
        return np.array(pil_img)  # RGB format
    except Exception as e:
        raise FileNotFoundError(f"Could not read {path}: {e}")


def get_dominant_colors(image, k=3, ignore_black=True, min_ratio=0.3):
    """Extract dominant colors using KMeans, ignore black and small/shadow clusters."""
    # resize for speed
    img = cv2.resize(image, (600, 400))
    img_flat = img.reshape((-1, 3))

    # run kmeans
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(img_flat)

    counts = Counter(labels)
    centers = kmeans.cluster_centers_

    # order by frequency
    ordered = counts.most_common()
    ordered_colors = [centers[i] for i, _ in ordered]

    # filter out black if requested
    if ignore_black:
        def is_black(c, threshold=40):  # v < 40 means black
            bgr = np.uint8([[c[::-1]]])
            h, s, v = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0][0]
            return v < threshold
        ordered_colors = [c for c in ordered_colors if not is_black(c)]

    # ---- merge similar colors ----
    merged_colors = []
    for idx, c in enumerate(ordered_colors):
        bgr = np.uint8([[c[::-1]]])
        h_raw, s, v = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0][0]
        h_deg = h_raw * 2
        hsv_c = (h_deg, int(s), int(v))

        merged = False
        for mc in merged_colors:
            if is_color_similar(hsv_c, mc["hsv"]):
                mc["count"] += counts[ordered[idx][0]]
                merged = True
                break

        if not merged:
            merged_colors.append({
                "rgb": tuple(map(int, c)),
                "hsv": hsv_c,
                "count": counts[ordered[idx][0]]
            })

    # ---- filter out small clusters (shadows, noise) ----
    total = sum(mc["count"] for mc in merged_colors)
    filtered_colors = [mc for mc in merged_colors if mc["count"] / total >= min_ratio]

    # safeguard: if all removed, keep the largest one
    if not filtered_colors and merged_colors:
        filtered_colors = [max(merged_colors, key=lambda mc: mc["count"])]

    hex_colors = [rgb_to_hex(mc["rgb"]) for mc in filtered_colors]

    return [mc["rgb"] for mc in filtered_colors], hex_colors

def increase_brightness(img, value=30):
    """Increase brightness of an RGB image by boosting V channel in HSV."""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img_bright = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
    return img_bright



def get_center_region(img, size=100):
    """
    擷取圖片的中央區域 (固定大小)。
    - img: 輸入圖片 (H, W, C)
    - size: 方形區域邊長 (像素)，預設 100
    - return: 中央裁切後的圖片
    """
    h, w = img.shape[:2]

    cx, cy = w // 2, h // 2  # 圖片中心點

    # 計算邊界
    x1 = max(cx - size // 2, 0)
    y1 = max(cy - size // 2, 0)
    x2 = min(cx + size // 2, w)
    y2 = min(cy + size // 2, h)

    return img[y1:y2, x1:x2]


# ===外型辨識函式 ===
def detect_shape_from_image(cropped_img, original_img=None, expected_shape=None):
    try:
        output = cropped_img.copy()
        thresh = preprocess_with_shadow_correction(output)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        shape = "其他"

        if not contours and original_img is not None:
            # print("⚠️ 無偵測到輪廓，改用原圖嘗試")#註解SSS
            gray_fallback = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
            _, thresh_fallback = cv2.threshold(gray_fallback, 127, 255, cv2.THRESH_BINARY)
            contours_fallback, _ = cv2.findContours(thresh_fallback, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours_fallback:
                main_contour = max(contours_fallback, key=cv2.contourArea)
                shape = detect_shape_three_classes(main_contour)
            else:
                print("⚠️ 二次嘗試仍無輪廓，標記為其他")  # 註解SSS
        elif contours:
            main_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(main_contour)
            img_area = cropped_img.shape[0] * cropped_img.shape[1]
            area_ratio = area / img_area
            # print(f"📐 輪廓面積：{area:.1f}，圖片面積：{img_area:.1f}，佔比：{area_ratio:.2%}")#註解SSS
            shape = detect_shape_three_classes(main_contour)

        if expected_shape:
            result = "✅" if shape == expected_shape else "❌"
            # print(f"📏 預測結果：{shape}，正確結果：{expected_shape} {result}")#註解SSS
            return shape, result
        return shape, None

    except Exception as e:
        print(f"❗ 發生錯誤：{e}")  # 註解SSS
        return "錯誤", None

def increase_brightness(img, value=30):
    """Increase brightness of an RGB image by boosting V channel in HSV."""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img_bright = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
    return img_bright

# === 增強處理函式 ===

def desaturate_image(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s.fill(0)
    return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)


def enhance_for_blur(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    contrast_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    blurred = cv2.GaussianBlur(contrast_img, (3, 3), 1.0)
    sharpened = cv2.addWeighted(contrast_img, 1.8, blurred, -0.8, 0)
    return cv2.bilateralFilter(sharpened, d=9, sigmaColor=75, sigmaSpace=75)


# === 3. 定義讀取 HEIC 的函式 ===


# === 形狀辨識相關 ===
def preprocess_with_shadow_correction(img_bgr):
    """校正陰影與自動二值化，改善輪廓品質"""
    # Step 1: 灰階轉換
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Step 2: 高斯模糊計算背景亮度
    blur = cv2.GaussianBlur(gray, (55, 55), 0)

    # Step 3: 灰階除以背景亮度 => 修正陰影
    corrected = cv2.divide(gray, blur, scale=255)

    # Step 4: 自適應 threshold => 適合局部亮度變化
    thresh = cv2.adaptiveThreshold(
        corrected, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 21, 10
    )

    return thresh


# === 放在檔案頂部定義統計用清單 ===
ratios_list = []


def detect_shape_three_classes(contour):
    shape = "其他"
    # print(len(contour))#註解SSS
    try:
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            (center, axes, angle) = ellipse
            major, minor = axes

            if minor == 0:
                return shape

            ratio = max(major, minor) / min(major, minor)
            ratios_list.append(ratio)
            # print(f"🔍 Ellipse ratio: {ratio:.3f}")#註解SSS

            # ➤ 分類
            if 0.95 <= ratio <= 1.15:
                shape = "圓形"
            elif ratio <= 2.3:
                shape = "橢圓形"
            else:
                shape = "其他"

            # print(f"📏 shape ratio: {ratio:.2f} => 判斷為 {shape}")

    except  Exception as e:
        print(f"❗ detect_shape_three_classes 發生錯誤：{e}")

    return shape
