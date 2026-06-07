import pyautogui
import cv2
import numpy as np
import time
import json
from skimage.metrics import structural_similarity
CONFIG_FILE = "game_config.json"
REGION_FILE = "regions.json"

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

    screenshot = pyautogui.screenshot(
        region=(left, top, right-left, bottom-top)
    )

    return cv2.cvtColor(
        np.array(screenshot),
        cv2.COLOR_RGB2BGR
    )

def similarity_score(img, template):
    result = cv2.matchTemplate(
        img,
        template,
        cv2.TM_CCOEFF_NORMED
    )
    return float(result.max())

def crop_region(image, region, debug=False):
    img_h, img_w = image.shape[:2]

    rx, ry, rw, rh = region

    x = int(rx * img_w)
    y = int(ry * img_h)
    w = int(rw * img_w)
    h = int(rh * img_h)

    if debug:
        print(
            f"x={x}, y={y}, w={w}, h={h}, "
            f"img_w={img_w}, img_h={img_h}"
        )

    crop = image[y:y+h, x:x+w]

    if debug:
        print("crop shape:", crop.shape)

    return crop

def compare_images(img1, img2):
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])

    img1 = cv2.resize(img1, (w, h))
    img2 = cv2.resize(img2, (w, h))

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    score, _ = structural_similarity(
        gray1,
        gray2,
        full=True
    )

    return score

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
        (cropped_icon3, cropped_label3)
    ]
def answer_phase(image,memory,debug=False):
    # Implement the logic for the answer phase here
    answer_icon = crop_region(image, ANSWER_ICON)
    cropped_answer1 = crop_region(image, ANSWER1)
    cropped_answer2 = crop_region(image, ANSWER2)
    cropped_answer3 = crop_region(image, ANSWER3)

    assert memory is not None, "Memory should be initialized as an empty dictionary if not provided."

    last_similarity_score = -1
    correct_icon = None

    for icon_img, label_img in memory:
        if icon_img is not None:
            score = compare_images(answer_icon, icon_img)
            if debug:
                print(f"Comparing answer icon with question icon, similarity score: {score:.4f}")
            if score > last_similarity_score:
                last_similarity_score = score
                correct_icon = icon_img
                correct_label = label_img
                if debug:
                    print(f"New best match found with score {score:.4f}")
                    cv2.imshow("Best Match", correct_icon)
                    cv2.waitKey(0)

    last_similarity_score = 0
    correct_answer = None

    for cropped_answer in cropped_answer1, cropped_answer2, cropped_answer3:
        score = compare_images(correct_label, cropped_answer)
        if debug:
            print(f"Comparing answer icon with the cropped answer, similarity score: {score:.4f}")
        if score > last_similarity_score:
            last_similarity_score = score
            correct_answer = cropped_answer
        if debug:
            print(f"New best match found with score {score:.4f}")
            cv2.imshow("Best Match", correct_answer)
            cv2.waitKey(0)

    cv2.imshow("Best Match", correct_answer)
    cv2.waitKey(0)

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
        
        #check what state it is
        answer_similarity_score = compare_images(RegionAnswer, ANSWER_SAMPLE)
        question_similarity_score = compare_images(RegionQuestion, QUESTION_SAMPLE)

        answer_match = answer_similarity_score > ANSWER_SIMILARITY_THRESHOLD
        question_match = question_similarity_score > QUESTION_SIMILARITY_THRESHOLD

        if not answer_match and not question_match:
            state = "not started"
        elif answer_match and not question_match:
            state = "answer"
        elif question_match and not answer_match:
            state = "question"
        else:
            # both matched
            if answer_similarity_score > question_similarity_score:
                state = "answer"
            else:
                state = "question"

        #we dont care if its the same stage
        if (state == last_state):
            print(f"Same as last state: {state}")
            time.sleep(1)
            continue
        
        #only record the information if its a new stage
        if (state == "question"):
            print("its the question phase")
            question_phase_data = question_phase(image)

        elif (state == "answer"):
            print("its the answer phase")
            answer_phase(image, memory=question_phase_data)
            
        else:
            print("its neither")

        last_state = state
        time.sleep(1)
        continue

run_game()