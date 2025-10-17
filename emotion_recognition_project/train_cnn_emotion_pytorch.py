import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ---------------------------
# Check device
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# ---------------------------
# Dataset
# ---------------------------
class EmotionDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.images = []
        self.labels = []
        self.transform = transform

        emotions = sorted(os.listdir(data_dir))
        self.emotion_to_idx = {emotion: idx for idx, emotion in enumerate(emotions)}

        for emotion in emotions:
            emotion_idx = self.emotion_to_idx[emotion]
            emotion_path = os.path.join(data_dir, emotion)
            for img_file in os.listdir(emotion_path):
                img_path = os.path.join(emotion_path, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # read grayscale
                img = cv2.resize(img, (48,48))
                self.images.append(img)
                self.labels.append(emotion_idx)

        self.images = np.array(self.images, dtype=np.float32) / 255.0  # normalize to 0-1
        self.images = self.images[:, np.newaxis, :, :]  # add channel dim: (N,1,48,48)
        self.labels = np.array(self.labels, dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return torch.tensor(img, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# ---------------------------
# CNN Model
# ---------------------------
class CNNEmotion(nn.Module):
    def __init__(self, num_classes):
        super(CNNEmotion, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 24x24
            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 12x12
            nn.Conv2d(64,128,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)   # 6x6
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(128*6*6, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

# ---------------------------
# Training function
# ---------------------------
def train_model(data_dir, epochs=30, batch_size=64, lr=0.001):
    dataset = EmotionDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    num_classes = len(os.listdir(data_dir))
    model = CNNEmotion(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0
        correct = 0
        total = 0

        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total * 100
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

    # Save model
    os.makedirs("../saved", exist_ok=True)
    torch.save(model.state_dict(), "../saved/cnn_emotion_pytorch.pth")
    print("Model saved to ../saved/cnn_emotion_pytorch.pth")

# ---------------------------
# Run training
# ---------------------------
if __name__ == "__main__":
    train_model("./emotion_dataset/train")
