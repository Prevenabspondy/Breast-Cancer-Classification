import os
import copy
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, classification_report


# ===================== CONFIG ===================== #

class Config:
    data_dir = "data"        # folder created by prepare_data.py
    num_classes = 2          # benign / malignant
    img_size = 224
    batch_size = 8           # smaller batch -> less RAM, faster per step
    num_epochs = 5           # changed to 5 epochs as requested
    lr = 1e-4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "efficientnet_cbam_breast_cancer.pth"


cfg = Config()


# ===================== DATA LOADERS ===================== #

def get_dataloaders():
    train_transforms = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        # You can comment these two lines out if still too slow
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_dir = os.path.join(cfg.data_dir, "train")
    val_dir = os.path.join(cfg.data_dir, "val")
    test_dir = os.path.join(cfg.data_dir, "test")

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_test_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transforms)

    # NOTE: num_workers=0 to avoid Windows hanging
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size,
                            shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size,
                             shuffle=False, num_workers=0)

    class_names = train_dataset.classes
    print("Classes:", class_names)

    dataloaders = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }

    dataset_sizes = {
        "train": len(train_dataset),
        "val": len(val_dataset),
        "test": len(test_dataset)
    }

    return dataloaders, dataset_sizes, class_names


# ===================== CBAM ATTENTION ===================== #

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        hidden = max(in_channels // reduction_ratio, 1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_channels, bias=False)
        )

    def forward(self, x):
        # x: (B, C, H, W)
        avg_pool = torch.mean(x, dim=(2, 3))         # (B, C)
        max_pool, _ = torch.max(x, dim=2)            # (B, C, W)
        max_pool, _ = torch.max(max_pool, dim=2)     # (B, C)

        avg_out = self.mlp(avg_pool)
        max_out = self.mlp(max_pool)
        out = avg_out + max_out
        scale = torch.sigmoid(out).unsqueeze(2).unsqueeze(3)  # (B, C, 1, 1)
        return x * scale


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=padding, bias=False)

    def forward(self, x):
        # x: (B, C, H, W)
        avg_out = torch.mean(x, dim=1, keepdim=True)        # (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)      # (B, 1, H, W)
        concat = torch.cat([avg_out, max_out], dim=1)       # (B, 2, H, W)
        attn = torch.sigmoid(self.conv(concat))             # (B, 1, H, W)
        return x * attn, attn


class CBAMBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.channel_att = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        x = self.channel_att(x)
        x, spatial_map = self.spatial_att(x)
        return x, spatial_map      # spatial_map is the attention map


# ===================== EFFICIENTNET + CBAM MODEL ===================== #

class EfficientNetCBAMNet(nn.Module):
    """
    EfficientNet-B0 backbone + CBAM on final feature map
    """
    def __init__(self, num_classes=2):
        super().__init__()

        backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )

        # EfficientNet structure: features -> avgpool -> classifier
        self.backbone_features = backbone.features   # Sequential
        in_channels = backbone.classifier[1].in_features  # 1280 for B0

        self.cbam = CBAMBlock(in_channels)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(in_channels, num_classes)

    def forward(self, x, return_attention=False):
        # Feature extraction
        feat = self.backbone_features(x)             # (B, C, H', W')

        # CBAM
        feat_att, spatial_map = self.cbam(feat)

        # Classification
        pooled = self.global_pool(feat_att)          # (B, C, 1, 1)
        pooled = torch.flatten(pooled, 1)            # (B, C)
        logits = self.classifier(pooled)             # (B, num_classes)

        if return_attention:
            # Upsample attention map to input resolution for visualization
            att_upsampled = F.interpolate(
                spatial_map, size=x.size()[2:], mode="bilinear",
                align_corners=False
            )
            return logits, att_upsampled
        else:
            return logits


# ===================== METRICS ===================== #

def calculate_metrics(y_true, y_pred):
    """
    Sensitivity = TP / (TP + FN)
    Specificity = TN / (TN + FP)
    """
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
    else:
        sensitivity = 0.0
        specificity = 0.0

    accuracy = np.mean(np.array(y_true) == np.array(y_pred))
    return accuracy, sensitivity, specificity


# ===================== TRAINING & EVAL ===================== #

def train_model(model, dataloaders, dataset_sizes, num_epochs=25, lr=1e-4):
    device = cfg.device
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 40)

        for phase in ["train", "val"]:
            if phase not in dataloaders:
                continue

            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            all_labels = []
            all_preds = []

            loader = dataloaders[phase]
            num_batches = len(loader)

            for batch_idx, (inputs, labels) in enumerate(loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

                # Progress print every few batches
                if batch_idx % 10 == 0 or batch_idx == num_batches - 1:
                    print(f"{phase} epoch {epoch+1}: batch {batch_idx+1}/{num_batches}")

            epoch_loss = running_loss / dataset_sizes[phase]
            acc, sens, spec = calculate_metrics(all_labels, all_preds)

            print(f"{phase} Loss: {epoch_loss:.4f} | "
                  f"Acc: {acc:.4f} | Sens: {sens:.4f} | Spec: {spec:.4f}")

            if phase == "val" and acc > best_val_acc:
                best_val_acc = acc
                best_model_wts = copy.deepcopy(model.state_dict())

    print(f"\nBest val Acc: {best_val_acc:.4f}")
    model.load_state_dict(best_model_wts)
    return model


def evaluate_model(model, dataloader, dataset_size, phase_name="test"):
    device = cfg.device
    model = model.to(device)
    model.eval()

    running_loss = 0.0
    all_labels = []
    all_preds = []

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * inputs.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    test_loss = running_loss / dataset_size
    acc, sens, spec = calculate_metrics(all_labels, all_preds)

    # ---- Additional evaluation outputs ----
    print(f"\n{phase_name} Loss: {test_loss:.4f} | "
          f"Acc: {acc:.4f} | Sens: {sens:.4f} | Spec: {spec:.4f}")

    # Confusion matrix and classification report
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)

    try:
        report = classification_report(all_labels, all_preds, digits=4)
        print("\nClassification Report:")
        print(report)
    except Exception as e:
        # If classification_report fails for some reason, still continue
        print("Could not produce classification report:", e)


# ===================== GRAD-CAM IMPLEMENTATION ===================== #

class GradCAM:
    """
    Simple Grad-CAM for a given target layer.
    Usage:
        target_layer = model.backbone_features[-1]
        gradcam = GradCAM(model, target_layer)
        cam = gradcam.generate(input_tensor)
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()

        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        """
        input_tensor: (1, C, H, W)
        returns: Grad-CAM heatmap resized to input size (H, W), numpy array
        """
        device = cfg.device
        input_tensor = input_tensor.to(device)

        self.model.zero_grad()

        outputs = self.model(input_tensor)  # logits
        if class_idx is None:
            class_idx = outputs.argmax(dim=1).item()

        loss = outputs[:, class_idx].sum()
        loss.backward()

        gradients = self.gradients
        activations = self.activations

        # Global average pooling on gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # (B, C, 1, 1)

        # Weighted sum of activations
        cam = torch.sum(weights * activations, dim=1)   # (B, H', W')
        cam = F.relu(cam)

        # Normalize
        cam = cam[0]
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        # Upsample to input size
        cam = cam.unsqueeze(0).unsqueeze(0)  # (1, 1, H', W')
        cam_up = F.interpolate(cam, size=input_tensor.size()[2:],
                               mode="bilinear", align_corners=False)
        cam_up = cam_up.squeeze().cpu().numpy()  # (H, W)
        return cam_up


# ===================== VISUALIZATION (CBAM + GRAD-CAM) ===================== #

def visualize_explanations(model, gradcam, img_path, class_names,
                           save_path="example_cbam_gradcam.png"):
    """
    1. Original image
    2. CBAM attention map overlay
    3. Grad-CAM overlay
    """
    from PIL import Image

    device = cfg.device
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    image = Image.open(img_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    # ---- CBAM ----
    with torch.no_grad():
        logits, cbam_att = model(input_tensor, return_attention=True)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        pred_idx = logits.argmax(dim=1).item()
        pred_class = class_names[pred_idx]

    cbam_map = cbam_att.squeeze().cpu().numpy()
    cbam_map = (cbam_map - cbam_map.min()) / (cbam_map.max() - cbam_map.min() + 1e-8)

    # ---- Grad-CAM ----
    cam = gradcam.generate(input_tensor, class_idx=pred_idx)

    # ---- Plot ----
    img_np = np.array(image.resize((cfg.img_size, cfg.img_size)))

    plt.figure(figsize=(12, 4))

    # Original
    plt.subplot(1, 3, 1)
    plt.imshow(img_np)
    plt.title("Original")
    plt.axis("off")

    # CBAM overlay
    plt.subplot(1, 3, 2)
    plt.imshow(img_np)
    plt.imshow(cbam_map, cmap="jet", alpha=0.4)
    plt.title("CBAM attention")
    plt.axis("off")

    # Grad-CAM overlay
    plt.subplot(1, 3, 3)
    plt.imshow(img_np)
    plt.imshow(cam, cmap="jet", alpha=0.4)
    title = f"Pred: {pred_class}\n"
    for i, cls in enumerate(class_names):
        title += f"{cls}: {probs[i]:.2f} "
    plt.title(title)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"Saved explanation visualization to {save_path}")
    plt.close()


# ===================== MAIN ===================== #

def main():
    dataloaders, dataset_sizes, class_names = get_dataloaders()

    model = EfficientNetCBAMNet(num_classes=cfg.num_classes)

    print("\nStarting training...")
    model = train_model(
        model,
        dataloaders,
        dataset_sizes,
        num_epochs=cfg.num_epochs,
        lr=cfg.lr
    )

    # Save best model
    torch.save(model.state_dict(), cfg.model_path)
    print(f"Model saved to {cfg.model_path}")

    # Evaluate on test set
    evaluate_model(model, dataloaders["test"], dataset_sizes["test"],
                   phase_name="test")

    # Create Grad-CAM object (target last EfficientNet feature block)
    target_layer = model.backbone_features[-1]   # last MBConv block
    gradcam = GradCAM(model, target_layer)

    # Example visualization on one test image
    class0_dir = os.path.join(cfg.data_dir, "test", class_names[0])
    example_img = os.path.join(class0_dir, os.listdir(class0_dir)[0])
    print("Generating explanation for:", example_img)

    visualize_explanations(
        model,
        gradcam,
        example_img,
        class_names,
        save_path="example_cbam_gradcam1.png"
    )


if __name__ == "__main__":
    main()
