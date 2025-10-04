# PROJECT_AI/train_cnn_full.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from cnn_model import AntiSpoofCNN  # Bạn cần có file cnn_model.py với class AntiSpoofCNN

# --- 1. Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# --- 2. Transforms ---
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 3 channels RGB
])

# --- 3. Dataset & DataLoader ---
train_data = datasets.ImageFolder("datasets/antispoof/train", transform=transform)
val_data   = datasets.ImageFolder("datasets/antispoof/val", transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_data, batch_size=32)

print(f"[INFO] Number of training images: {len(train_data)}")
print(f"[INFO] Number of validation images: {len(val_data)}")

# --- 4. Model, Loss, Optimizer ---
model = AntiSpoofCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- 5. Training ---
num_epochs = 65

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)

    # --- Validation ---
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    val_acc = 100 * correct / total

    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f} - Val Acc: {val_acc:.2f}%")

# --- 6. Save model ---
torch.save(model.state_dict(), "cnn_face_antispoof.pth")
print("[INFO] Training finished. Model saved as cnn_face_antispoof.pth")
