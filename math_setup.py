import cv2
import json

CONFIG_FILE = "math_config.json"
REGION_FILE = "math_regions.json"


def select_regions(img_path, region_names):
    img = cv2.imread(img_path)

    if img is None:
        raise FileNotFoundError(img_path)

    h, w = img.shape[:2]

    regions = {}

    for name in region_names:
        print(f"Select {name}")

        x, y, rw, rh = cv2.selectROI(
            name,
            img,
            showCrosshair=True,
            fromCenter=False
        )

        cv2.destroyWindow(name)

        regions[name] = [
            round(x / w, 4),
            round(y / h, 4),
            round(rw / w, 4),
            round(rh / h, 4)
        ]

    return regions


def setup():
    with open(CONFIG_FILE, "r") as f:
        config = json.load(f)

    question_image = config["question_image"]
    answer_image = config["answer_image"]

    regions = {}

    regions.update(
        select_regions(
            question_image,
            [
                "question_state_roi",
                "question",
            ]
        )
    )

    regions.update(
        select_regions(
            answer_image,
            [
                "answer_state_roi",
                "jump_1_button",
                "jump_5_button",
                "next_button",
            ]
        )
    )

    with open(REGION_FILE, "w") as f:
        json.dump(regions, f, indent=4)

    print(f"Saved {REGION_FILE}")


if __name__ == "__main__":
    setup()
