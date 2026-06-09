import time
import cv2
import pyautogui
import numpy as np
import pytesseract
import difflib
from pathlib import Path

def get_game_window(window_title):
    windows = pyautogui.getWindowsWithTitle(window_title)
    if not windows:
        print(f"ERROR: Could not find a window with title '{window_title}'.")
        exit()
    win = windows[0]
    return win

def activate_windows(win):
    if win.isMinimized:
        win.restore()
        while win.isMinimized:
            time.sleep(0.05)
    if not win.isActive:
        win.activate()
        time.sleep(0.2)

def screenshot_game(window_title):
    win = get_game_window(window_title)
    activate_windows(win)
    left, top = win.topleft
    right, bottom = win.bottomright
    screenshot = pyautogui.screenshot(region=(left, top, right-left, bottom-top))
    return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

def click_region(window_title, region, debug=False):
    win = get_game_window(window_title)
    rx, ry, rw, rh = region
    x = win.left + int((rx + rw / 2) * win.width)
    y = win.top  + int((ry + rh / 2) * win.height)
    if debug:
        print(f"Clicking ({x}, {y})")
    pyautogui.click(x, y)

def crop_region(image, region, debug=False):
    img_h, img_w = image.shape[:2]
    rx, ry, rw, rh = region
    x = int(rx * img_w)
    y = int(ry * img_h)
    w = int(rw * img_w)
    h = int(rh * img_h)
    if debug:
        print(f"x={x}, y={y}, w={w}, h={h}, img_w={img_w}, img_h={img_h}")
    crop = image[y:y+h, x:x+w]
    return crop

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def split_figure_masks(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, np.array([90, 40, 80]), np.array([130, 255, 255]))
    pink = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([140, 40, 80]), np.array([180, 255, 255])),
        cv2.inRange(hsv, np.array([0, 40, 80]), np.array([15, 255, 255])),
    )
    return blue, pink

def clean_mask(mask):
    if cv2.countNonZero(mask) == 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

def preprocess_icon(img):
    blue, pink = split_figure_masks(img)
    mask = cv2.bitwise_or(blue, pink)
    if cv2.countNonZero(mask) < 20:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return clean_mask(mask)

def normalize_icon_mask(mask, size=64):
    pts = cv2.findNonZero(mask)
    if pts is None:
        return np.zeros((size, size), dtype=np.uint8)
    x, y, w, h = cv2.boundingRect(pts)
    crop = mask[y:y + h, x:x + w]
    if crop.size == 0:
        return np.zeros((size, size), dtype=np.uint8)
    scale = size / max(w, h)
    new_w = max(int(w * scale), 1)
    new_h = max(int(h * scale), 1)
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size), dtype=np.uint8)
    ox = (size - new_w) // 2
    oy = (size - new_h) // 2
    canvas[oy:oy + new_h, ox:ox + new_w] = resized
    return canvas

def largest_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)

def hu_contour_similarity(mask1, mask2):
    c1 = largest_contour(mask1)
    c2 = largest_contour(mask2)
    if c1 is None or c2 is None:
        return 0.0
    if cv2.contourArea(c1) < 10 or cv2.contourArea(c2) < 10:
        return 0.0
    dist = cv2.matchShapes(c1, c2, cv2.CONTOURS_MATCH_I1, 0.0)
    return float(1.0 / (1.0 + dist))

def mask_iou(mask1, mask2):
    union = cv2.bitwise_or(mask1, mask2)
    intersection = cv2.bitwise_and(mask1, mask2)
    union_count = cv2.countNonZero(union)
    if union_count == 0:
        return 1.0
    return cv2.countNonZero(intersection) / union_count

def figure_centroid(mask):
    moments = cv2.moments(mask)
    if moments["m00"] == 0:
        return None
    return np.array([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]])

def normalized_figure_positions(blue_mask, pink_mask):
    combined = cv2.bitwise_or(blue_mask, pink_mask)
    pts = cv2.findNonZero(combined)
    if pts is None:
        return None, None
    x, y, w, h = cv2.boundingRect(pts)

    def norm_centroid(mask):
        centroid = figure_centroid(mask)
        if centroid is None:
            return None
        return np.array([(centroid[0] - x) / max(w, 1), (centroid[1] - y) / max(h, 1)])

    return norm_centroid(blue_mask), norm_centroid(pink_mask)

def color_area_ratio(mask_blue, mask_pink):
    blue_area = cv2.countNonZero(mask_blue)
    pink_area = cv2.countNonZero(mask_pink)
    total = blue_area + pink_area
    if total == 0:
        return None
    return np.array([blue_area / total, pink_area / total])

def edge_chamfer_similarity(mask1, mask2):
    n1 = normalize_icon_mask(mask1)
    n2 = normalize_icon_mask(mask2)
    e1 = cv2.Canny(n1, 40, 120)
    e2 = cv2.Canny(n2, 40, 120)
    if cv2.countNonZero(e1) == 0 or cv2.countNonZero(e2) == 0:
        return 0.0
    dt = cv2.distanceTransform(cv2.bitwise_not(e2), cv2.DIST_L2, 3)
    pts = cv2.findNonZero(e1)
    distances = [float(dt[y, x]) for x, y in pts[:, 0]]
    return max(0.0, 1.0 - (sum(distances) / len(distances)) / 15.0)

def pose_layout_similarity(img1, img2):
    blue1, pink1 = split_figure_masks(img1)
    blue2, pink2 = split_figure_masks(img2)
    blue1 = clean_mask(blue1)
    pink1 = clean_mask(pink1)
    blue2 = clean_mask(blue2)
    pink2 = clean_mask(pink2)

    blue_sim = similarity_score(normalize_icon_mask(blue1), normalize_icon_mask(blue2))
    pink_sim = similarity_score(normalize_icon_mask(pink1), normalize_icon_mask(pink2))

    nb1, np1 = normalized_figure_positions(blue1, pink1)
    nb2, np2 = normalized_figure_positions(blue2, pink2)

    vec_sim = 0.0
    pos_sim = 0.0
    if nb1 is not None and np1 is not None and nb2 is not None and np2 is not None:
        v1 = np1 - nb1
        v2 = np2 - nb2
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 > 1e-3 and n2 > 1e-3:
            vec_sim = max(0.0, float(np.dot(v1 / n1, v2 / n2)))
            if v1[1] * v2[1] < 0:
                vec_sim *= 0.25
        pos_dist = np.linalg.norm(np1 - np2) + np.linalg.norm(nb1 - nb2)
        pos_sim = max(0.0, 1.0 - pos_dist / 2.0)

    ratio_sim = 0.0
    r1 = color_area_ratio(blue1, pink1)
    r2 = color_area_ratio(blue2, pink2)
    if r1 is not None and r2 is not None:
        ratio_sim = max(0.0, 1.0 - float(np.linalg.norm(r1 - r2)))

    return max(0.0, (
        0.30 * blue_sim
        + 0.30 * pink_sim
        + 0.20 * vec_sim
        + 0.10 * pos_sim
        + 0.10 * ratio_sim
    ))

def similarity_score(img1, img2):
    """
    Safely handles shape alignment, empty/black mask conditions,
    and returns the maximum template matching correlation value.
    """
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])

    if h == 0 or w == 0:
        return 0.0

    img1_resized = cv2.resize(img1, (w, h))
    img2_resized = cv2.resize(img2, (w, h))

    if len(img1_resized.shape) == 2:
        nz1 = cv2.countNonZero(img1_resized)
        nz2 = cv2.countNonZero(img2_resized)
    else:
        nz1 = 1 if np.any(img1_resized) else 0
        nz2 = 1 if np.any(img2_resized) else 0

    if nz1 == 0 and nz2 == 0:
        return 1.0
    if nz1 == 0 or nz2 == 0:
        return 0.0

    result = cv2.matchTemplate(img1_resized, img2_resized, cv2.TM_CCOEFF_NORMED)
    return float(result.max())

def label_gray_inverted(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    if float(gray.mean()) < 127:
        gray = 255 - gray
    return gray

def default_label_upscale(min_dim):
    if min_dim >= 64:
        return 1
    return max(2, (64 + min_dim - 1) // min_dim)

def preprocess_label_at_scale(img, scale):
    """Prepare a label crop at a specific upscale factor."""
    if img is None or img.size == 0:
        return None

    gray = label_gray_inverted(img)
    h, w = gray.shape[:2]
    if scale > 1:
        gray = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.copyMakeBorder(thresh, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)

def preprocess_label_for_ocr(img):
    """Prepare a small white-on-dark label crop for Tesseract."""
    if img is None or img.size == 0:
        return None
    min_dim = min(img.shape[:2])
    return preprocess_label_at_scale(img, default_label_upscale(min_dim))

def preprocess_label_for_ocr_alt(img):
    """Alternate threshold for difficult digit crops."""
    if img is None or img.size == 0:
        return None

    gray = label_gray_inverted(img)
    h, w = gray.shape[:2]
    scale = default_label_upscale(min(h, w))
    if scale > 1:
        gray = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return cv2.copyMakeBorder(thresh, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)

def digit_image_similarity(label_img, answer_img):
    """Compare two digit crops visually when OCR is unreliable."""
    a = preprocess_label_at_scale(label_img, 1)
    b = preprocess_label_at_scale(answer_img, 1)
    if a is None or b is None:
        return 0.0
    return similarity_score(a, b)

def ocr_best_confidence(img, config, digits_only=False):
    best_text = ""
    best_conf = -1
    try:
        data = pytesseract.image_to_data(
            img, config=config, output_type=pytesseract.Output.DICT
        )
    except pytesseract.TesseractError:
        data = None

    if data:
        for text, conf in zip(data["text"], data["conf"]):
            text = text.strip()
            conf = int(conf)
            if conf <= 0 or not text:
                continue
            if digits_only:
                digits = "".join(ch for ch in text if ch.isdigit())
                if not digits:
                    continue
                text = digits[0]
            if conf > best_conf:
                best_conf = conf
                best_text = text

    if not best_text:
        try:
            fallback = pytesseract.image_to_string(img, config=config).strip()
        except pytesseract.TesseractError:
            fallback = ""
        if digits_only:
            digits = "".join(ch for ch in fallback if ch.isdigit())
            if len(digits) == 1:
                return digits, 50
            return "", -1
        if fallback:
            return fallback, 50

    return best_text, best_conf

def try_ocr_digit_on_image(prep):
    best_text = ""
    best_conf = -1
    for config in (
        "--psm 10 -c tessedit_char_whitelist=0123456789",
        "--psm 8 -c tessedit_char_whitelist=0123456789",
    ):
        text, conf = ocr_best_confidence(prep, config, digits_only=True)
        if text and conf > best_conf:
            best_text, best_conf = text, conf
    return best_text, best_conf

def read_digit_ocr(img):
    """Fast single-digit OCR with multi-scale fallback for glyphs like 0."""
    if img is None or img.size == 0:
        return ""

    min_dim = min(img.shape[:2])
    default_scale = default_label_upscale(min_dim)
    scales = []
    for scale in (default_scale, 1, 3):
        if scale not in scales:
            scales.append(scale)

    best_text = ""
    best_conf = -1
    for scale in scales:
        prep = preprocess_label_at_scale(img, scale)
        if prep is None:
            continue
        text, conf = try_ocr_digit_on_image(prep)
        if text and conf > best_conf:
            best_text, best_conf = text, conf
        if text and conf >= 60:
            return text

    alternate = preprocess_label_for_ocr_alt(img)
    if alternate is not None:
        text, conf = try_ocr_digit_on_image(alternate)
        if text and conf > best_conf:
            best_text = text

    return best_text

def read_label_ocr(img, digits_only=False):
    """Read text from a cropped UI label."""
    if digits_only:
        return read_digit_ocr(img)

    primary = preprocess_label_for_ocr(img)
    if primary is None:
        return ""

    whitelist = "0123456789+-*/xX() "
    text, conf = ocr_best_confidence(
        primary, f"--psm 7 -c tessedit_char_whitelist={whitelist}"
    )
    if text and conf >= 50:
        return text.strip()

    alt_text, alt_conf = ocr_best_confidence(
        primary, f"--psm 6 -c tessedit_char_whitelist={whitelist}"
    )
    if alt_conf > conf and alt_text:
        text = alt_text
    return text.strip()

def icon_similarity_details(img1, img2):
    mask1 = preprocess_icon(img1)
    mask2 = preprocess_icon(img2)
    norm1 = normalize_icon_mask(mask1)
    norm2 = normalize_icon_mask(mask2)

    template_score = similarity_score(norm1, norm2)
    layout_score = pose_layout_similarity(img1, img2)
    contour_score = hu_contour_similarity(mask1, mask2)
    iou_score = mask_iou(norm1, norm2)
    edge_score = edge_chamfer_similarity(mask1, mask2)
    total = max(0.0, (
        0.20 * template_score
        + 0.35 * layout_score
        + 0.15 * contour_score
        + 0.15 * iou_score
        + 0.15 * edge_score
    ))
    return {
        "total": float(total),
        "template": float(template_score),
        "layout": float(layout_score),
        "contour": float(contour_score),
        "iou": float(iou_score),
        "edge": float(edge_score),
    }

def icon_similarity(img1, img2):
    return icon_similarity_details(img1, img2)["total"]

def text_similarity(str1, str2):
    """Performs a fuzzy string comparison to protect against minor OCR misreads."""
    if not str1 or not str2:
        return 0.0
    return difflib.SequenceMatcher(None, str1.lower(), str2.lower()).ratio()

def state_detection(image, question_sample, answer_sample, question_state_roi, answer_state_roi,
                    question_threshold, answer_threshold, debug=False):
    region_question = crop_region(image, question_state_roi)
    region_answer = crop_region(image, answer_state_roi)

    answer_similarity_score = similarity_score(preprocess(region_answer), preprocess(answer_sample))
    question_similarity_score = similarity_score(preprocess(region_question), preprocess(question_sample))

    if debug:
        print(f"Answer similarity: {answer_similarity_score}")
        print(f"Question similarity: {question_similarity_score}")

    answer_match = answer_similarity_score > answer_threshold
    question_match = question_similarity_score > question_threshold

    if not answer_match and not question_match:
        state = "not started"
    elif answer_match and not question_match:
        state = "answer"
    elif question_match and not answer_match:
        state = "question"
    else:
        if answer_similarity_score > question_similarity_score:
            state = "answer"
        else:
            state = "question"
    return state

def save_image_pair(folder, name, img, is_icon=False):
    raw_path = folder / f"{name}_raw.png"
    thresh_path = folder / f"{name}_thresholded.png"
    cv2.imwrite(str(raw_path), img)

    if is_icon:
        cv2.imwrite(str(thresh_path), preprocess_icon(img))
    else:
        cv2.imwrite(str(thresh_path), preprocess(img))

    return {"raw": str(raw_path), "thresholded": str(thresh_path)}
