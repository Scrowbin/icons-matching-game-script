import pyautogui
import cv2
import numpy as np
import time
import json

CONFIG_FILE = "game_config.json"
with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)
WINDOW_TITLE = config.get("game_name", "")
SIMILARITY_THRESHOLD = config.get("confidence_threshold", 0.96)
QUESTION_IMAGE = cv2.imread(config.get("question_image", "sample/question.png"))
ANSWER_IMAGE = cv2.imread(config.get("answer_image", "sample/answer.png"))

def get_game_window():
    windows = pyautogui.getWindowsWithTitle(WINDOW_TITLE)
    if not windows:
        print(f"ERROR: Could not find a window with title '{WINDOW_TITLE}'.")
        exit()
    win = windows[0]

    return win

def activate_windows(win):
    win.restore()

    while win.isMinimized:
        time.sleep(0.05)
    win.activate()
    time.sleep(0.2)

# def setup_phase():
#     if os.path.exists(CONFIG_FILE):
#        print("There already exits a config file")
#        pass

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

def run_game():
    while True:
        image = screenshot_game()
        state = "not started"
        if (similarity_score(image, QUESTION_IMAGE)>SIMILARITY_THRESHOLD):
            state = "question"
        elif (similarity_score(image, ANSWER_IMAGE)>SIMILARITY_THRESHOLD):
            state = "answer"
        else:
            state = "not started"

        
        if (state == "question"):
            print("its the question phase")


            #implement later
        elif (state == "answer"):
            print("its the answer phase")
            #implement later
        else:
            print("its neither")
        time.sleep(1)
        continue

run_game()