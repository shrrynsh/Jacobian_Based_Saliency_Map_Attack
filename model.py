import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):

    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(50 * 4 * 4, 500)
        self.fc2 = nn.Linear(500, num_classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        return x.view(x.size(0), -1)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.logits(x), dim=1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(x).argmax(dim=1)

    def predict_logits(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.logits(x).argmax(dim=1)

    @staticmethod
    def load_model(path: str, device: torch.device = None) -> "LeNet5":
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = LeNet5().to(device)
        state = torch.load(path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        return model
