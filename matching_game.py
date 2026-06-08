import pytesseract
import time
import json
import os
from pathlib import Path
from helper import *

DEBUG_MODE = True
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
CONFIG_FILE = "matching_game.json"
REGION_FILE = "matching_regions.json"
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
batch_counter = 0

with open(CONFIG_FILE, 'r') as f:
    config = json.load(f)
WINDOW_TITLE = config.get("game_name", "")
QUESTION_SIMILARITY_THRESHOLD = config.get("question_confidence_threshold", 0.85)
ANSWER_SIMILARITY_THRESHOLD = config.get("answer_confidence_threshold", 0.87)
QUESTION_IMAGE = cv2.imread(config.get("question_image", "sample/match_question.png"))
ANSWER_IMAGE = cv2.imread(config.get("answer_image", "sample/match_answer.png"))

with open(REGION_FILE, 'r') as f:
    region_config = json.load(f)

QUESTION_STATE_ROI = region_config.get("question_state_roi", [0, 0, 0, 0])
QUESTION_ICON1 = region_config.get("question_icon_1", [0, 0, 0, 0])
QUESTION_ICON2 = region_config.get("question_icon_2", [0, 0, 0, 0])
QUESTION_ICON3 = region_config.get("question_icon_3", [0, 0, 0, 0])
QUESTION_LABEL1 = region_config.get("question_label_1", [0, 0, 0, 0])
QUESTION_LABEL2 = region_config.get("question_label_2", [0, 0, 0, 0])
QUESTION_LABEL3 = region_config.get("question_label_3", [0, 0, 0, 0])
ANSWER_STATE_ROI = region_config.get("answer_state_roi", [0, 0, 0, 0])
ANSWER_ICON = region_config.get("answer_icon", [0, 0, 0, 0])
ANSWER1 = region_config.get("answer_1", [0, 0, 0, 0])
ANSWER2 = region_config.get("answer_2", [0, 0, 0, 0])
ANSWER3 = region_config.get("answer_3", [0, 0, 0, 0])


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

    mask_answer = preprocess_icon(answer_icon)

    for i, (icon_img, _) in enumerate(memory):
        if icon_img is not None:
            mask_question = preprocess_icon(icon_img)
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
    question_sample = crop_region(QUESTION_IMAGE, QUESTION_STATE_ROI)
    answer_sample = crop_region(ANSWER_IMAGE, ANSWER_STATE_ROI)
    question_phase_data = []

    while True:
        image = screenshot_game(WINDOW_TITLE)
        state = state_detection(
            image, question_sample, answer_sample,
            QUESTION_STATE_ROI, ANSWER_STATE_ROI,
            QUESTION_SIMILARITY_THRESHOLD, ANSWER_SIMILARITY_THRESHOLD
        )

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
                click_region(WINDOW_TITLE, answer_regions[answer_idx])
        else:
            print("its neither")

        last_state = state
        time.sleep(1)

if __name__ == "__main__":
    run_game()
