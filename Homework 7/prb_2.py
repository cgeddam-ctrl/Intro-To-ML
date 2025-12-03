import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import time


# =========================================================
# GPU OPTIMIZATIONS
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True   # Auto-tune best conv kernels
print("Using device:", device)


# =========================================================
# DATA TRANSFORMS
# =========================================================
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])


# =========================================================
# SIMPLE FAST CNN MODEL (Modify to your needs)
# =========================================================
class FastCNN(nn.Module):
    def __init__(self):
        super(FastCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# =========================================================
# TRAINING FUNCTION (with AMP & tqdm)
# =========================================================
def train(model, loader, optimizer, criterion, scaler):
    model.train()
    running_loss = 0.0

    for inputs, targets in tqdm(loader, desc="Training"):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        # Mixed precision
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * inputs.size(0)

    return running_loss / len(loader.dataset)


# =========================================================
# TEST FUNCTION
# =========================================================
def test(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return 100 * correct / total


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    # FAST DATALOADERS FOR RTX 4070
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=256, shuffle=True,
        num_workers=0, pin_memory=True
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=512, shuffle=False,
        num_workers=0, pin_memory=True
    )

    # Build model
    model = FastCNN().to(device)

    # Loss + Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    # Automatic Mixed Precision
    scaler = GradScaler()

    num_epochs = 200  # With 4070, 50 epochs takes ~3 minutes
    start_time = time.time()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        train_loss = train(model, trainloader, optimizer, criterion, scaler)
        acc = test(model, testloader)

        print(f"Loss: {train_loss:.4f} | Test Acc: {acc:.2f}%")

    print(f"\nTotal Training Time: {time.time() - start_time:.2f} seconds")
