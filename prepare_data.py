import os
import shutil
import random

RAW_DIR = "raw_data"   # has benign/ and malignant/
OUT_DIR = "data"       # will contain train/val/test/
SPLIT = (0.7, 0.15, 0.15)  # train, val, test
CLASSES = ["benign", "malignant"]  # change if you used Benigno/Maligno
SEED = 42


def make_dir(path):
    os.makedirs(path, exist_ok=True)


def split_and_copy():
    random.seed(SEED)

    # create output structure
    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            make_dir(os.path.join(OUT_DIR, split, cls))

    for cls in CLASSES:
        src_dir = os.path.join(RAW_DIR, cls)
        files = [f for f in os.listdir(src_dir)
                 if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".bmp"))]

        random.shuffle(files)

        n_total = len(files)
        n_train = int(SPLIT[0] * n_total)
        n_val = int(SPLIT[1] * n_total)
        n_test = n_total - n_train - n_val

        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:]

        print(f"{cls}: total={n_total}, train={len(train_files)}, "
              f"val={len(val_files)}, test={len(test_files)}")

        def copy_files(file_list, split_name):
            for fname in file_list:
                src = os.path.join(src_dir, fname)
                dst = os.path.join(OUT_DIR, split_name, cls, fname)
                shutil.copy2(src, dst)

        copy_files(train_files, "train")
        copy_files(val_files, "val")
        copy_files(test_files, "test")

    print("Done! Data split into train/val/test in the 'data' folder.")


if __name__ == "__main__":
    split_and_copy()
