import cv2
import json

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
    regions = {}

    regions.update(
        select_regions(
            "sample/question.png",
            [
                "question_state_roi",  # NEW
                "question_icon_1",
                "question_label_1",
                "question_icon_2",
                "question_label_2",
                "question_icon_3",
                "question_label_3"
            ]
        )
    )

    regions.update(
        select_regions(
            "sample/answer.png",
            [
                "answer_state_roi",    # NEW
                "answer_icon",
                "answer_1",
                "answer_2",
                "answer_3"
            ]
        )
    )

    with open("regions.json", "w") as f:
        json.dump(regions, f, indent=4)

    print("Saved regions.json")


if __name__ == "__main__":
    setup()