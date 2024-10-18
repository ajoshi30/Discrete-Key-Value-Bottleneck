import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Transformations for CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# CIFAR-10 Dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
class KeyValueBottleneck(nn.Module):
    def __init__(self, encoder, num_keys=100, key_dim=128, value_dim=128):
        super(KeyValueBottleneck, self).__init__()
        self.encoder = encoder
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.num_keys = num_keys
        
        # Key-Value codebook
        self.keys = nn.Parameter(torch.randn(num_keys, key_dim))
        self.values = nn.Parameter(torch.randn(num_keys, value_dim))

        # Decoder (simple linear classifier)
        self.decoder = nn.Linear(value_dim, 10)

    def forward(self, x):
        encoded = self.encoder(x)
        encoded = encoded.view(encoded.size(0), -1)  # Flatten

        # Compute similarity to keys
        similarities = torch.matmul(encoded, self.keys.T)
        nearest_key_index = similarities.argmax(dim=1)

        # Fetch corresponding value
        value = self.values[nearest_key_index]
        
        # Pass the value to the decoder
        out = self.decoder(value)
        return out
from torchvision import models

# Load a pre-trained ResNet50 model as the encoder
encoder = models.resnet50(pretrained=True)
encoder = nn.Sequential(*list(encoder.children())[:-1])  # Remove final classification layer

# Initialize the key-value bottleneck model
model = KeyValueBottleneck(encoder, num_keys=400, key_dim=2048, value_dim=256).cuda()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
def train(model, train_loader, optimizer, criterion, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.cuda(), labels.cuda()
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%')

def test(model, test_loader, criterion):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

# Train and test the model
train(model, train_loader, optimizer, criterion, num_epochs=10)
test(model, test_loader, criterion)

