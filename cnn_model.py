# PROJECT_AI/cnn_model.py
import torch
import torch.nn as nn

class AntiSpoofCNN(nn.Module):
    def __init__(self):
        super(AntiSpoofCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*16*16, 256), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)  # 2 classes: real, fake
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
