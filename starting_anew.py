import pyautogui
import cv2
import numpy as np
import time
import json
import os
import difflib
import pytesseract
from pathlib import Path

# Global Debug Toggle
DEBUG_MODE = True
#hardcoded, too lazy to edit path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
CONFIG_FILE = "game_config.json"
REGION_FILE = "regions.json"
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
batch_counter = 0

with open(CONFIG_FILE, 'r') as f:
    config = json.load(f)
WINDOW_TITLE = config.get("game_name", "")
QUESTION_SIMILARITY_THRESHOLD = config.get("question_confidence_threshold", 0.85)
ANSWER_SIMILARITY_THRESHOLD = config.get("answer_confidence_threshold", 0.87)
QUESTION_IMAGE = cv2.imread(config.get("question_image", "sample/question.png"))
ANSWER_IMAGE = cv2.imread(config.get("answer_image", "sample/answer.png"))

with open(REGION_FILE, 'r') as f:
    region_config = json.load(f)

QUESTION_ICON1 = region_config.get("question_icon_1", [0, 0, 0, 0])
QUESTION_ICON2 = region_config.get("question_icon_2", [0, 0, 0, 0])
QUESTION_ICON3 = region_config.get("question_icon_3", [0, 0, 0, 0])      

QUESTION_LABEL1 = region_config.get("question_label_1", [0, 0, 0, 0])
QUESTION_LABEL2 = region_config.get("question_label_2", [0, 0, 0, 0])
QUESTION_LABEL3 = region_config.get("question_label_3", [0, 0, 0, 0])

ANSWER_ICON = region_config.get("answer_icon", [0, 0, 0, 0])
ANSWER1 = region_config.get("answer_1", [0, 0, 0, 0])
ANSWER2 = region_config.get("answer_2", [0, 0, 0, 0])
ANSWER3 = region_config.get("answer_3", [0, 0, 0, 0])

def get_game_window():
    windows = pyautogui.getWindowsWithTitle(WINDOW_TITLE)
    if not windows:
        print(f"ERROR: Could not find a window with title '{WINDOW_TITLE}'.")
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

def screenshot_game():
    win = get_game_window()
    activate_windows(win)
    left, top = win.topleft
    right, bottom = win.bottomright
    screenshot = pyautogui.screenshot(region=(left, top, right-left, bottom-top))
    return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

def click_region(region, debug=False):
    win = get_game_window()
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

def preprocess_icon(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([90, 20, 100])
    upper = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    return mask

def similarity_score(img1, img2):
    """
    Safely handles shape alignment, empty/black mask conditions, 
    and returns the maximum template matching correlation value.
    """
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])

    if h == 0 or w == 0:
        return 0.0

    # Align dimensions dynamically
    img1_resized = cv2.resize(img1, (w, h))
    img2_resized = cv2.resize(img2, (w, h))

    # Safeguard against completely empty/black frames (prevents matchTemplate zero-division crash)
    if len(img1_resized.shape) == 2:
        nz1 = cv2.countNonZero(img1_resized)
        nz2 = cv2.countNonZero(img2_resized)
    else:
        nz1 = 1 if np.any(img1_resized) else 0
        nz2 = 1 if np.any(img2_resized) else 0

    if nz1 == 0 and nz2 == 0:
        return 1.0  # Both elements are empty / identical blanks
    if nz1 == 0 or nz2 == 0:
        return 0.0  # One contains an item, the other is completely blank

    result = cv2.matchTemplate(img1_resized, img2_resized, cv2.TM_CCOEFF_NORMED)
    return float(result.max())

def read_label_ocr(img):
    """Uses Tesseract OCR to read text labels from cropped UI regions."""
    thresh = preprocess(img)
    # PSM 7 treats the image as a single text line, ideal for game labels/digits
    text = pytesseract.image_to_string(thresh, config='--psm 7').strip()
    return text

def text_similarity(str1, str2):
    """Performs a fuzzy string comparison to protect against minor OCR misreads."""
    if not str1 and not str2:
        return 1.0
    if not str1 or not str2:
        return 0.0
    return difflib.SequenceMatcher(None, str1.lower(), str2.lower()).ratio() # <--- CORRECT

def save_image_pair(folder, name, img, is_icon=False):
    raw_path = folder / f"{name}_raw.png"
    thresh_path = folder / f"{name}_thresholded.png"
    cv2.imwrite(str(raw_path), img)
    
    if is_icon:
        cv2.imwrite(str(thresh_path), preprocess_icon(img))
    else:
        cv2.imwrite(str(thresh_path), preprocess(img))

    return {"raw": str(raw_path), "thresholded": str(thresh_path)}

def question_phase(image, debug=False):
    global batch_counter
    cropped_icon1 = crop_region(image, QUESTION_ICON1)
    cropped_icon2 = crop_region(image, QUESTION_ICON2)
    cropped_icon3 = crop_region(image, QUESTION_ICON3)
    cropped_label1 = crop_region(image, QUESTION_LABEL1)
    cropped_label2 = crop_region(image, QUESTION_LABEL2)
    cropped_label3 = crop_region(image, QUESTION_LABEL3)

    question_phase_data = [
        (cropped_icon1, cropped_label1),
        (cropped_icon2, cropped_label2),
        (cropped_icon3, cropped_label3)
    ]

    if debug:
        batch_counter += 1
        batch_dir = Path(LOG_DIR) / f"batch_{batch_counter:04d}"
        batch_dir.mkdir(parents=True, exist_ok=True)

        question_log = {"icons": [], "labels": []}
        for i, (icon, label) in enumerate(question_phase_data, start=1):
            question_log["icons"].append(save_image_pair(batch_dir, f"question_icon_{i}", icon, is_icon=True))
            question_log["labels"].append(save_image_pair(batch_dir, f"question_label_{i}", label, is_icon=False))
        return question_phase_data, question_log, batch_dir
    
    return question_phase_data

def answer_phase(image, memory, batch_dir, question_log, debug=False):
    answer_icon = crop_region(image, ANSWER_ICON)
    cropped_answer1 = crop_region(image, ANSWER1)
    cropped_answer2 = crop_region(image, ANSWER2)
    cropped_answer3 = crop_region(image, ANSWER3)

    assert memory is not None, "Memory should be initialized."

    icon_scores = []
    best_icon_score = -1
    best_icon_idx = None

    # Step 1: Turn the answer icon into a binary shape profile using HSV
    mask_answer = preprocess_icon(answer_icon)

    for i, (icon_img, _) in enumerate(memory):
        if icon_img is not None:
            # Step 2: Extract the binary shape profile of the saved question icon
            mask_question = preprocess_icon(icon_img)

            # Step 3: Compare shapes directly by passing the binary masks into similarity_score
            score = similarity_score(mask_answer, mask_question)
            
            icon_scores.append({
                "question_icon": i + 1,
                "score": float(score)
            })
            
            if debug:
                print(f"Comparing shapes of answer mask with question mask {i+1}, score: {score:.4f}")
            
            if score > best_icon_score:
                best_icon_score = score
                best_icon_idx = i

    # --- AMBIGUITY CHECK ---
    if len(icon_scores) >= 2:
        sorted_icon_scores = sorted([s["score"] for s in icon_scores], reverse=True)
        margin = sorted_icon_scores[0] - sorted_icon_scores[1]
        
        if debug:
            print(f"Icon Shape match margin (1st vs 2nd): {margin:.4f}")
            
        if margin < 0.05:
            print("Icon shape overlap too ambiguous (margin < 0.05), skipping click.")
            return None
    
    if debug:
        print(f"Best structural mask shape match found: {best_icon_idx + 1} with score {best_icon_score:.4f}")

    # Step 4: Extract the Text identity using OCR
    target_text = read_label_ocr(memory[best_icon_idx][1])
    if debug:
        print(f"Target memory string read via OCR: '{target_text}'")

    answers = [cropped_answer1, cropped_answer2, cropped_answer3]
    answer_scores = []
    best_idx = None
    best_score = -1

    for i, answer in enumerate(answers):
        option_text = read_label_ocr(answer)
        score = text_similarity(target_text, option_text)
        
        if debug:
            print(f"Option {i+1} OCR text: '{option_text}' | String similarity score: {score:.4f}")

        answer_scores.append({
            "answer": i + 1,
            "detected_text": option_text,
            "score": float(score)
        })

        if score > best_score:
            best_score = score
            best_idx = i

    if debug:
        answer_log = {}
        answer_log["answer_icon"] = save_image_pair(batch_dir, "answer_icon", answer_icon, is_icon=True)

        for i, ans in enumerate(answers, start=1):
            answer_log[f"answer_{i}"] = save_image_pair(batch_dir, f"answer_{i}", ans, is_icon=False)

        log_entry = {
            "timestamp": time.time(),
            "matched_question_icon": best_icon_idx + 1 if best_icon_idx is not None else None,
            "target_ocr_text": target_text,
            "icon_shape_similarity": float(best_icon_score),
            "icon_margin": float(margin) if 'margin' in locals() else None,
            "answer_scores": answer_scores,
            "selected_answer": best_idx + 1 if best_idx is not None else None,
            "selected_answer_score": float(best_score),
            "question_files": question_log,
            "answer_files": answer_log
        }
        with open(batch_dir / "decision.json", "w") as f:
            json.dump(log_entry, f, indent=4)
            
    if best_score < 0.5:
        print("Text match validation confidence too low, skipping click.")
        return None
    
    print(f"Selected answer {best_idx + 1} (text match score={best_score:.4f})")
    return best_idx

def run_game():
    state = "not started"
    last_state = None
    QUESTION_SAMPLE = crop_region(QUESTION_IMAGE, region_config.get("question_state_roi", [0, 0, 0, 0]))
    ANSWER_SAMPLE = crop_region(ANSWER_IMAGE, region_config.get("answer_state_roi", [0, 0, 0, 0]))

    question_phase_data = []

    while True:
        image = screenshot_game()
        RegionQuestion = crop_region(image, region_config.get("question_state_roi", [0, 0, 0, 0]))
        RegionAnswer = crop_region(image, region_config.get("answer_state_roi", [0, 0, 0, 0]))
        
        answer_similarity_score = similarity_score(preprocess(RegionAnswer), preprocess(ANSWER_SAMPLE))
        question_similarity_score = similarity_score(preprocess(RegionQuestion), preprocess(QUESTION_SAMPLE))

        answer_match = answer_similarity_score > ANSWER_SIMILARITY_THRESHOLD
        question_match = question_similarity_score > QUESTION_SIMILARITY_THRESHOLD

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

        if (state == last_state):
            time.sleep(1)
            continue
        
        if (state == "question"):
            print("its the question phase")
            if not DEBUG_MODE:
                question_phase_data = question_phase(image)
            else:
                question_phase_data, question_log, batch_dir = question_phase(image, debug=True)

        elif (state == "answer"):
            print("its the answer phase")
            if DEBUG_MODE:
                answer_idx = answer_phase(image, question_phase_data, batch_dir, question_log, debug=True)
            else:
                answer_idx = answer_phase(image, question_phase_data, None, None, debug=False)

            if answer_idx is not None:
                answer_regions = [ANSWER1, ANSWER2, ANSWER3]
                click_region(answer_regions[answer_idx])
        else:
            print("its neither")

        last_state = state
        time.sleep(1)

run_game()