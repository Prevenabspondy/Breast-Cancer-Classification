import os
# import uuid

# from flask import Flask, render_template, request, redirect, url_for
import gradio as gr
import tempfile
import torch
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Import your model, GradCAM, and Config from the training file
from breast_cancer_efficientnet_cbam_gradcam import EfficientNetCBAMNet, GradCAM, Config

print("========== APP STARTED ==========")

print("Loading Config...")


# ---------------- CONFIG ---------------- #

cfg = Config()
DEVICE = cfg.device  # "cuda" or "cpu"
MODEL_PATH = cfg.model_path  # efficientnet_cbam_breast_cancer.pth
CLASS_NAMES = ["benign", "malignant"]  # must match training order

print("Config Loaded")



# ---------------- LOAD MODEL ---------------- #

print("Loading model...")
model = EfficientNetCBAMNet(num_classes=cfg.num_classes)
print("Model Created")
print("Loading weights...")
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
print("Weights Loaded")
print("Loading state_dict into model...")
model.load_state_dict(state_dict)
print("state_dict Loaded")
print("Moving model to device...")
model.to(DEVICE)
print("Model moved")
print("Setting model to eval...")
model.eval()
print("Model Ready")

# Grad-CAM target layer: last EfficientNet feature block
print("Creating GradCAM...")
target_layer = model.backbone_features[-1]
gradcam = GradCAM(model, target_layer)
print("GradCAM Ready")

# Transform similar to validation / test
transform = transforms.Compose([
    transforms.Resize((cfg.img_size, cfg.img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---------------- UTILS ---------------- #


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


# ---------------- GRADIO FUNCTION ---------------- #

def predict(image):
    if image is None:
        return None, "Please upload an image."

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_input:
        image.save(temp_input.name)
        input_path = temp_input.name

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_output:
        result_path = temp_output.name

    diagnosis, probs = generate_explanation(input_path, result_path)

    confidence = ""
    for i, cls in enumerate(CLASS_NAMES):
        confidence += f"{cls.capitalize()}: {probs[i] * 100:.2f}%\n"

    with Image.open(result_path) as img:
        result_img = img.copy()

    # Delete temporary files
    os.remove(input_path)
    os.remove(result_path)

    return result_img, f"Prediction: {diagnosis.upper()}\n\n{confidence}"


# ---------------- GRADIO UI ---------------- #
print("Creating Gradio UI...")

with gr.Blocks(title="Breast Cancer Detection") as demo:

    gr.Markdown("# Breast Cancer Detection using EfficientNet-B0 + CBAM + Grad-CAM")

    with gr.Row():

        input_image = gr.Image(type="pil", label="Upload Image")

        output_image = gr.Image(label="CBAM + Grad-CAM")

    output_text = gr.Textbox(
        label="Prediction",
        lines=6
    )

    predict_btn = gr.Button("Predict")

    predict_btn.click(
        fn=predict,
        inputs=input_image,
        outputs=[output_image, output_text]
    )

port = int(os.environ.get("PORT", 7860))
print("Launching Gradio...")
print("Gradio Started")
demo.launch(
    server_name="0.0.0.0",
    server_port=port
)
