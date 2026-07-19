import os
import shutil

# ====== CHANGE THIS TO YOUR REAL PATH ======
# Example: r"D:\datasets\classificacao_binaria"
BREAKHIS_BINARY_ROOT = r"D:\datasets\classificacao_binaria"
# ===========================================

OUT_RAW = "raw_data"          # where we put all images for training

# Adjust magnification folder names if needed
magnifications = ["40X", "100X", "200X", "400X"]

# Adjust these if your folders are named differently
classes = ["benign", "malignant"]
# If your folders are "Benigno" and "Maligno", use:
# classes = ["Benigno", "Maligno"]


def main():
    # make output folders
    for cls in classes:
        os.makedirs(os.path.join(OUT_RAW, cls), exist_ok=True)

    for mag in magnifications:
        for cls in classes:
            src_dir = os.path.join(BREAKHIS_BINARY_ROOT, mag, cls)
            dst_dir = os.path.join(OUT_RAW, cls)

            if not os.path.isdir(src_dir):
                print(f"WARNING: folder not found: {src_dir}")
                continue

            print(f"Copying from {src_dir} -> {dst_dir}")
            for fname in os.listdir(src_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".bmp")):
                    src = os.path.join(src_dir, fname)
                    # add magnification prefix to avoid duplicate names
                    dst = os.path.join(dst_dir, f"{mag}_{fname}")
                    shutil.copy2(src, dst)

    print("Finished! Check the folder: raw_data")


if __name__ == "__main__":
    main()
