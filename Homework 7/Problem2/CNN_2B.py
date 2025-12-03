import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
print("Using device:", device)

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

# DATALOADERS 
def get_loaders(batch_size=256):
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=1000, shuffle=False, num_workers=0
    )
    return trainloader, testloader

# BASIC RESIDUAL BLOCK 
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, use_dropout=False):
        super(BasicBlock, self).__init__()

        self.use_dropout = use_dropout
        if use_dropout:
            self.drop = nn.Dropout(p=0.3)

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # Shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))

        if self.use_dropout:
            out = self.drop(out)

        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return torch.relu(out)

# RESNET-10 Model
class ResNet10(nn.Module):
    def __init__(self, use_dropout=False):
        super(ResNet10, self).__init__()
        self.in_planes = 16
        self.use_dropout = use_dropout

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(16, stride=1)
        self.layer2 = self._make_layer(32, stride=2)
        self.layer3 = self._make_layer(64, stride=2)
        self.layer4 = self._make_layer(128, stride=2)

        self.linear = nn.Linear(128, 10)

    def _make_layer(self, planes, stride):
        block = BasicBlock(self.in_planes, planes, stride,
                           use_dropout=self.use_dropout)
        self.in_planes = planes
        return block

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = torch.mean(out, dim=[2, 3])
        out = self.linear(out)
        return out

# TRAINING AND TESTING LOOPS 
def train_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss = 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)

def test_epoch(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return 100.0 * correct / total

# FUNCTION TO RUN 1 EXPERIMENT
def run_experiment(name, use_dropout=False, weight_decay=0.0):
    print(f"\n Running Experiment: {name}")

    trainloader, testloader = get_loaders()

    model = ResNet10(use_dropout=use_dropout).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=0.01, momentum=0.9, weight_decay=weight_decay
    )

    scaler = GradScaler()
    num_epochs = 200

    start_time = time.time()

    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        train_loss = train_epoch(model, trainloader, optimizer, criterion, scaler)
        test_acc = test_epoch(model, testloader)
        print(f"Loss: {train_loss:.4f} | Test Accuracy: {test_acc:.2f}%")

    total_time = time.time() - start_time

    print("\n========== FINAL RESULTS ==========")
    print(f"Model: {name}")
    print(f" Training Time: {total_time:.2f} sec")
    print(f" Final Training Loss: {train_loss:.4f}")
    print(f"Final Test Accuracy: {test_acc:.2f}%\n")

    return total_time, train_loss, test_acc

# MAIN: RUN 3 REGULARIZATION EXPERIMENTS
if __name__ == "__main__":

    wd_results = run_experiment(
        "Weight Decay λ=0.001",
        use_dropout=False,
        weight_decay=0.001
    )

    dropout_results = run_experiment(
        "Dropout p=0.3",
        use_dropout=True,
        weight_decay=0.0
    )

    bn_results = run_experiment(
        "BatchNorm Only (Same as 2a)",
        use_dropout=False,
        weight_decay=0.0
    )

    print("\n\n FINAL SUMMARY TABLE ")
    print("Model | Time (s) | Loss | Accuracy")
    print(f"Weight Decay | {wd_results[0]:.2f} | {wd_results[1]:.4f} | {wd_results[2]:.2f}%")
    print(f"Dropout | {dropout_results[0]:.2f} | {dropout_results[1]:.4f} | {dropout_results[2]:.2f}%")
    print(f"BatchNorm (Baseline) | {bn_results[0]:.2f} | {bn_results[1]:.4f} | {bn_results[2]:.2f}%")
