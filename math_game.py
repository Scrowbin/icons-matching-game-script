import pytesseract
from helper import *
import os
import json
import re

DEBUG_MODE = True
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
CONFIG_FILE = "math_config.json"
REGION_FILE = "math_regions.json"
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
batch_counter = 0

with open(CONFIG_FILE, 'r') as f:
    config = json.load(f)
WINDOW_TITLE = config.get("game_name", "")
QUESTION_SIMILARITY_THRESHOLD = config.get("question_confidence_threshold", 0.85)
ANSWER_SIMILARITY_THRESHOLD = config.get("answer_confidence_threshold", 0.87)
QUESTION_IMAGE = cv2.imread(config.get("question_image", "sample/math_question.png"))
ANSWER_IMAGE = cv2.imread(config.get("answer_image", "sample/math_answer.png"))

with open(REGION_FILE, 'r') as f:
    region_config = json.load(f)

QUESTION_STATE_ROI = region_config.get("question_state_roi", [0, 0, 0, 0])
QUESTION = region_config.get("question", [0, 0, 0, 0])
ANSWER_STATE_ROI = region_config.get("answer_state_roi", [0, 0, 0, 0])
JUMP_1_BUTTON = region_config.get("jump_1_button", [0, 0, 0, 0])
JUMP_5_BUTTON = region_config.get("jump_5_button", [0, 0, 0, 0])
NEXT_BUTTON = region_config.get("next_button", [0, 0, 0, 0])


def parse_and_solve_math(ocr_text):
    """
    Cleans an OCR string and safely evaluates basic arithmetic expressions.
    """
    cleaned = ocr_text.lower().replace('x', '*')
    cleaned = re.sub(r'[^0-9\+\-\*\/\(\)\s]', '', cleaned)
    cleaned = cleaned.strip()
    if not cleaned:
        return None

    try:
        result = eval(cleaned)
        return int(round(result))
    except Exception as e:
        print(f"[-] Math evaluation failed for parsed text '{cleaned}': {e}")
        return None

def question_phase(image, debug=False):
    question_crop = crop_region(image, QUESTION)
    readlabel = read_label_ocr(question_crop)
    result = parse_and_solve_math(readlabel)
    if debug:
        print(f"Question OCR: '{readlabel}'\nParsed Result: {result}")

    return result

def answer_phase(result, debug=False):
    num_press_jump_1 = result % 5
    num_press_jump_5 = result // 5
    if debug:
        print(f"Calculated jumps: {num_press_jump_5} x 5-jumps and {num_press_jump_1} x 1-jumps")

    for _ in range(num_press_jump_1):
        click_region(WINDOW_TITLE, JUMP_1_BUTTON, debug=debug)
        time.sleep(0.2)
    for _ in range(num_press_jump_5):
        click_region(WINDOW_TITLE, JUMP_5_BUTTON, debug=debug)
        time.sleep(0.2)
    click_region(WINDOW_TITLE, NEXT_BUTTON, debug=debug)

def run_game():
    state = "not started"
    question_sample = crop_region(QUESTION_IMAGE, QUESTION_STATE_ROI)
    answer_sample = crop_region(ANSWER_IMAGE, ANSWER_STATE_ROI)
    last_state = None
    question = None

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
            question = question_phase(image, debug=DEBUG_MODE)

        elif (state == "answer"):
            print("its the answer phase")
            answer_phase(question, debug=DEBUG_MODE)

        else:
            print("its neither")

        last_state = state
        time.sleep(1)

if __name__ == "__main__":
    run_game()
