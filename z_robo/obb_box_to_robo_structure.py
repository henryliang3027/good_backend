import os
import zipfile

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(BASE_DIR)

IMAGES_DIR = os.path.join(REPO_DIR, "original_box_images")
LABELS_DIR = os.path.join(REPO_DIR, "label_obb_boxes")
OUTPUT_ZIP = os.path.join(BASE_DIR, "obb_box_dataset.zip")

CLASSES = ["a0", "a1", "a2", "b0", "c0", "d0", "e0", "f0", "f1", "g0", "h0"]


def main():
    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in sorted(os.listdir(IMAGES_DIR)):
            zf.write(os.path.join(IMAGES_DIR, fname), fname)

        for fname in sorted(os.listdir(LABELS_DIR)):
            zf.write(os.path.join(LABELS_DIR, fname), fname)

        zf.writestr("classes.txt", "\n".join(CLASSES) + "\n")

    print(f"Wrote {OUTPUT_ZIP}")


if __name__ == "__main__":
    main()
