import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# ---------------- TRANSFORMS ----------------
train_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((256,256)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(224, scale=(0.8,1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485]*3, std=[0.229]*3)
])

val_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485]*3, std=[0.229]*3)
])

# ---------------- DATA ----------------
train_data = ImageFolder("dataset/train", transform=train_transforms)
val_data = ImageFolder("dataset/val", transform=val_transforms)

train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
val_loader = DataLoader(val_data, batch_size=8, shuffle=False)

num_classes = len(train_data.classes)
print("Classes:", train_data.classes)

# ---------------- MODEL ----------------
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.to(device)

# ---------------- TRAINING ----------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

epochs = 25

for epoch in range(epochs):
    model.train()
    running_loss = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    model.eval()
    correct = total = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss:.3f} | Val Acc: {acc:.2f}%")

# from sklearn.metrics import confusion_matrix
# import numpy as np

# model.eval()
# y_true, y_pred = [], []

# with torch.no_grad():
#     for imgs, labels in val_loader:
#         imgs = imgs.to(device)
#         outputs = model(imgs)
#         preds = outputs.argmax(1).cpu().numpy()
#         y_pred.extend(preds)
#         y_true.extend(labels.numpy())

# cm = confusion_matrix(y_true, y_pred)
# print("Confusion Matrix:\n", cm)

# ---------------- SAVE MODEL ----------------
torch.save(model.state_dict(), "xray_defect_model.pth")
print("Model saved as xray_defect_model.pth")
