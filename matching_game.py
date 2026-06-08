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

ICON_TEXT_MATCH_MIN = 0.12
ICON_TEXT_MARGIN_MIN = 0.02
TEXT_AMBIGUITY_MARGIN = 0.03
ICON_SELECT_MARGIN = 0.20


def initial_batch_counter():
    existing = [
        int(path.name.split("_")[1])
        for path in Path(LOG_DIR).glob("batch_*")
        if path.is_dir() and path.name.split("_")[1].isdigit()
    ]
    return max(existing, default=0)


def question_phase(image):
    cropped_icon1 = crop_region(image, QUESTION_ICON1)
    cropped_icon2 = crop_region(image, QUESTION_ICON2)
    cropped_icon3 = crop_region(image, QUESTION_ICON3)
    cropped_label1 = crop_region(image, QUESTION_LABEL1)
    cropped_label2 = crop_region(image, QUESTION_LABEL2)
    cropped_label3 = crop_region(image, QUESTION_LABEL3)

    return [
        (cropped_icon1, cropped_label1),
        (cropped_icon2, cropped_label2),
        (cropped_icon3, cropped_label3),
    ]

def question_capture_quality(memory):
    icon_pixels = [cv2.countNonZero(preprocess_icon(icon)) for icon, _ in memory]
    if not icon_pixels or min(icon_pixels) < 15:
        return 0
    return sum(icon_pixels)

def build_question_log(batch_dir, memory):
    question_log = {"icons": [], "labels": []}
    for i, (icon, label) in enumerate(memory, start=1):
        question_log["icons"].append(save_image_pair(batch_dir, f"question_icon_{i}", icon, is_icon=True))
        question_log["labels"].append(save_image_pair(batch_dir, f"question_label_{i}", label, is_icon=False))
    return question_log

def score_icon_answer_pairs(answer_icon, memory, answers):
    icon_scores = []
    for icon_idx, (icon_img, _) in enumerate(memory):
        icon_details = icon_similarity_details(answer_icon, icon_img)
        icon_scores.append({
            "question_icon": icon_idx + 1,
            "icon_score": icon_details["total"],
            "icon_details": icon_details,
        })
    icon_scores.sort(key=lambda row: -row["icon_score"])
    icon_shape_margin = (
        icon_scores[0]["icon_score"] - icon_scores[1]["icon_score"]
        if len(icon_scores) > 1 else icon_scores[0]["icon_score"]
    )
    selected_icons = (
        {icon_scores[0]["question_icon"]}
        if icon_shape_margin >= ICON_SELECT_MARGIN
        else {row["question_icon"] for row in icon_scores}
    )

    pairs = []
    for icon_idx, (icon_img, label_img) in enumerate(memory):
        icon_details = icon_similarity_details(answer_icon, icon_img)
        icon_score = icon_details["total"]
        label_text = read_label_ocr(label_img)
        for answer_idx, answer_img in enumerate(answers):
            answer_text = read_label_ocr(answer_img)
            text_score = text_similarity(label_text, answer_text)
            combined = icon_score * (0.25 + 0.75 * text_score)
            pairs.append({
                "question_icon": icon_idx + 1,
                "answer": answer_idx + 1,
                "icon_score": float(icon_score),
                "icon_details": icon_details,
                "label_text": label_text,
                "answer_text": answer_text,
                "text_score": float(text_score),
                "combined": float(combined),
                "icon_locked": (icon_idx + 1) in selected_icons,
            })

    eligible = [row for row in pairs if row["icon_locked"]]
    eligible.sort(key=lambda row: row["combined"], reverse=True)
    pairs.sort(key=lambda row: row["combined"], reverse=True)
    return pairs, eligible[0], icon_shape_margin

def skip_reason_for_pairs(pairs):
    best = pairs[0]
    second = pairs[1] if len(pairs) > 1 else None
    margin = best["combined"] - second["combined"] if second else best["combined"]

    if best["text_score"] >= 1.0 and best["label_text"] and best["answer_text"]:
        for other in pairs[1:]:
            if other["text_score"] >= 1.0 and (best["combined"] - other["combined"]) < TEXT_AMBIGUITY_MARGIN:
                return f"Ambiguous perfect text matches (margin < {TEXT_AMBIGUITY_MARGIN})"
        return None

    if second and margin < ICON_TEXT_MARGIN_MIN:
        return f"Match too ambiguous (margin < {ICON_TEXT_MARGIN_MIN})"

    if best["combined"] < ICON_TEXT_MATCH_MIN:
        return f"Match confidence too low ({best['combined']:.4f})"

    return None

def write_decision_log(batch_dir, question_log, answer_log, pairs, best, margin,
                       icon_shape_margin=0.0, skipped=False, skip_reason=None):
    log_entry = {
        "timestamp": time.time(),
        "skipped": skipped,
        "skip_reason": skip_reason,
        "matched_question_icon": best["question_icon"],
        "target_ocr_text": best["label_text"],
        "icon_shape_similarity": best["icon_score"],
        "icon_margin": float(margin),
        "icon_shape_margin": float(icon_shape_margin),
        "pair_scores": pairs,
        "selected_answer": None if skipped else best["answer"],
        "selected_answer_score": best["text_score"],
        "combined_score": best["combined"],
        "question_files": question_log,
        "answer_files": answer_log,
    }
    with open(batch_dir / "decision.json", "w") as f:
        json.dump(log_entry, f, indent=4)

def answer_phase(image, memory, batch_dir, question_log, debug=False):
    answer_icon = crop_region(image, ANSWER_ICON)
    answers = [
        crop_region(image, ANSWER1),
        crop_region(image, ANSWER2),
        crop_region(image, ANSWER3),
    ]

    assert memory is not None, "Memory should be initialized."

    pairs, best, icon_shape_margin = score_icon_answer_pairs(answer_icon, memory, answers)
    eligible = [row for row in pairs if row["icon_locked"]]
    margin = (
        eligible[0]["combined"] - eligible[1]["combined"]
        if len(eligible) > 1 else eligible[0]["combined"]
    )
    skip_reason = skip_reason_for_pairs(eligible)

    if debug:
        for row in pairs:
            print(
                f"Pair icon={row['question_icon']} ans={row['answer']}: "
                f"icon={row['icon_score']:.4f}, text={row['text_score']:.4f}, "
                f"label={row['label_text']!r}, option={row['answer_text']!r}, "
                f"combined={row['combined']:.4f}"
            )
        print(f"Icon shape margin (1st vs 2nd icon): {icon_shape_margin:.4f}")
        print(f"Best pair margin (1st vs 2nd): {margin:.4f}")

        answer_log = {"answer_icon": save_image_pair(batch_dir, "answer_icon", answer_icon, is_icon=True)}
        for i, ans in enumerate(answers, start=1):
            answer_log[f"answer_{i}"] = save_image_pair(batch_dir, f"answer_{i}", ans, is_icon=False)

        write_decision_log(
            batch_dir, question_log, answer_log, pairs, best, margin, icon_shape_margin,
            skipped=skip_reason is not None,
            skip_reason=skip_reason,
        )

    if skip_reason:
        print(f"{skip_reason}, skipping click.")
        return None

    print(
        f"Selected icon {best['question_icon']} / answer {best['answer']} "
        f"(combined={best['combined']:.4f}, text={best['text_score']:.4f})"
    )
    return best["answer"] - 1

def run_game():
    last_state = None
    question_sample = crop_region(QUESTION_IMAGE, QUESTION_STATE_ROI)
    answer_sample = crop_region(ANSWER_IMAGE, ANSWER_STATE_ROI)
    question_phase_data = []
    question_ready = False
    best_question_quality = 0
    batch_counter = initial_batch_counter()
    batch_dir = None
    question_log = None

    while True:
        image = screenshot_game(WINDOW_TITLE)
        state = state_detection(
            image, question_sample, answer_sample,
            QUESTION_STATE_ROI, ANSWER_STATE_ROI,
            QUESTION_SIMILARITY_THRESHOLD, ANSWER_SIMILARITY_THRESHOLD
        )

        if state == "question":
            candidate = question_phase(image)
            quality = question_capture_quality(candidate)
            if quality >= best_question_quality:
                question_phase_data = candidate
                best_question_quality = quality
            question_ready = True

        elif state == "answer" and state != last_state:
            if not question_ready:
                print("No question data captured for this round, skipping.")
            else:
                if DEBUG_MODE:
                    batch_counter += 1
                    batch_dir = Path(LOG_DIR) / f"batch_{batch_counter:04d}"
                    batch_dir.mkdir(parents=True, exist_ok=True)
                    question_log = build_question_log(batch_dir, question_phase_data)
                    answer_idx = answer_phase(image, question_phase_data, batch_dir, question_log, debug=True)
                else:
                    answer_idx = answer_phase(image, question_phase_data, None, None, debug=False)

                if answer_idx is not None:
                    answer_regions = [ANSWER1, ANSWER2, ANSWER3]
                    click_region(WINDOW_TITLE, answer_regions[answer_idx])
                question_ready = False
                best_question_quality = 0

        if state != last_state:
            if state == "question":
                print("its the question phase")
            elif state == "answer":
                print("its the answer phase")
            else:
                print("its neither")
                best_question_quality = 0

        last_state = state
        time.sleep(1)

if __name__ == "__main__":
    run_game()
