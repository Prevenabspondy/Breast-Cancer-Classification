import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os

from breast_cancer_efficientnet_cbam_gradcam import EfficientNetCBAMNet, GradCAM, Config

cfg = Config()
device = cfg.device


model = EfficientNetCBAMNet(num_classes=cfg.num_classes)
model.load_state_dict(torch.load(cfg.model_path, map_location=device))
model = model.to(device)
model.eval()


image_path = "test_image.jpg"   


transform = transforms.Compose([
    transforms.Resize((cfg.img_size, cfg.img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)

# ---- Prediction ----
with torch.no_grad():
    outputs, cbam_att = model(input_tensor, return_attention=True)
    probs = F.softmax(outputs, dim=1)[0].cpu().numpy()
    pred_idx = outputs.argmax(dim=1).item()
    classes = ["benign", "malignant"]
    pred_class = classes[pred_idx]

print(f"Prediction: {pred_class}")
print(f"Probabilities: {probs}")

# ---- Grad-CAM ----
target_layer = model.backbone_features[-1]
gradcam = GradCAM(model, target_layer)
cam = gradcam.generate(input_tensor, class_idx=pred_idx)

# ---- Show explanation ----
img_np = np.array(image.resize((cfg.img_size, cfg.img_size)))
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(img_np); plt.title("Original"); plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(img_np); plt.imshow(cbam_att.squeeze().cpu(), cmap="jet", alpha=0.4)
plt.title("CBAM"); plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(img_np); plt.imshow(cam, cmap="jet", alpha=0.4)
plt.title(f"Grad-CAM: {pred_class}"); plt.axis("off")

plt.show()
