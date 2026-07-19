import os
import uuid

from flask import Flask, render_template, request, redirect, url_for
import torch
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Import your model, GradCAM, and Config from the training file
from breast_cancer_efficientnet_cbam_gradcam import EfficientNetCBAMNet, GradCAM, Config

# ---------------- CONFIG ---------------- #

cfg = Config()
DEVICE = cfg.device  # "cuda" or "cpu"
MODEL_PATH = cfg.model_path  # efficientnet_cbam_breast_cancer.pth
CLASS_NAMES = ["benign", "malignant"]  # must match training order

UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "static/results"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# ---------------- FLASK APP ---------------- #

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ---------------- LOAD MODEL ---------------- #

print("Loading model...")
model = EfficientNetCBAMNet(num_classes=cfg.num_classes)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()

# Grad-CAM target layer: last EfficientNet feature block
target_layer = model.backbone_features[-1]
gradcam = GradCAM(model, target_layer)

# Transform similar to validation / test
transform = transforms.Compose([
    transforms.Resize((cfg.img_size, cfg.img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# ---------------- UTILS ---------------- #

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def generate_explanation(input_image_path: str, result_image_path: str):
    """
    Loads image, runs model, creates CBAM + Grad-CAM visualization,
    returns predicted class + probabilities (numpy array).
    """
    # Load image
    image = Image.open(input_image_path).convert("RGB")
    img_resized = image.resize((cfg.img_size, cfg.img_size))
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # Forward pass with CBAM attention
    with torch.no_grad():
        logits, cbam_att = model(input_tensor, return_attention=True)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        pred_idx = int(probs.argmax())
        pred_class = CLASS_NAMES[pred_idx]

    # Normalize CBAM map
    cbam_map = cbam_att.squeeze().cpu().numpy()
    cbam_map = (cbam_map - cbam_map.min()) / (cbam_map.max() - cbam_map.min() + 1e-8)

    # Grad-CAM
    cam = gradcam.generate(input_tensor, class_idx=pred_idx)

    # Convert original image to numpy
    img_np = np.array(img_resized)

    # Text with diagnosis and confidence
    text = f"Diagnosis: {pred_class.upper()}\n"
    for i, cls in enumerate(CLASS_NAMES):
        text += f"{cls}: {probs[i]:.2f}  "

    # Plot and save
    plt.figure(figsize=(11, 4))

    # Original
    plt.subplot(1, 3, 1)
    plt.imshow(img_np)
    plt.title("Original")
    plt.axis("off")

    # CBAM
    plt.subplot(1, 3, 2)
    plt.imshow(img_np)
    plt.imshow(cbam_map, cmap="jet", alpha=0.4)
    plt.title("CBAM Attention")
    plt.axis("off")

    # Grad-CAM
    plt.subplot(1, 3, 3)
    plt.imshow(img_np)
    plt.imshow(cam, cmap="jet", alpha=0.4)
    plt.title("Grad-CAM")
    plt.text(5, 15, text, fontsize=9, color="white",
             bbox=dict(facecolor="black", alpha=0.6))
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(result_image_path, dpi=200)
    plt.close()

    return pred_class, probs


# ---------------- ROUTES ---------------- #

@app.route("/", methods=["GET", "POST"])
def index():
    diagnosis = None
    probs = None
    result_image = None

    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]

        if file.filename == "":
            return redirect(request.url)

        if file and allowed_file(file.filename):
            # unique file names
            input_filename = f"{uuid.uuid4().hex}.png"
            input_path = os.path.join(app.config["UPLOAD_FOLDER"], input_filename)

            result_filename = f"{uuid.uuid4().hex}.png"
            result_path = os.path.join(RESULT_FOLDER, result_filename)

            file.save(input_path)

            # generate explanation image + get prediction
            diagnosis, probs = generate_explanation(input_path, result_path)

            # 🔹 convert numpy array to list for Jinja template
            if probs is not None:
                probs = probs.tolist()

            result_image = result_filename  # for HTML

    return render_template(
        "index.html",
        diagnosis=diagnosis,
        probs=probs,
        class_names=CLASS_NAMES,
        result_image=result_image
    )


if __name__ == "__main__":
    app.run(debug=True)
