import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

# ========== CONFIGURATION ==========
data_root = "./dataset"
batch_size = 32
num_epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== DATA LOADING ==========
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(root=os.path.join(data_root, 'train'), transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_path = os.path.join(data_root, 'val')
val_loader = DataLoader(
    datasets.ImageFolder(root=val_path, transform=transform),
    batch_size=batch_size, shuffle=False
)

# ========== MODEL ==========
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 3)  # 3 classes
model = model.to(device)

# ========== TRAINING ==========
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("📚 Starting Training...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# ========== VALIDATION ==========
print("\n🔍 Validating Model...")
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

accuracy = accuracy_score(all_labels, all_preds)
print(f"✅ Final Validation Accuracy: {accuracy * 100:.2f}%")

# ========== SAVE MODEL ==========
torch.save(model.state_dict(), "leaf_model.pth")
print("💾 Model saved to leaf_model.pth")



