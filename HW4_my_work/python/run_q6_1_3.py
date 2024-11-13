
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F  # a lower level (compared to torch.nn) interface
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
import scipy.io
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

# Define the network architecture
class Net(nn.Module):
    def __init__(self, num_class, H, W):
        super(Net, self).__init__()

        # Define the convolutional layers using nn.Sequential
        # First convolutional layer
        self.conv1 = nn.Sequential(
            # Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0,
            #        dilation=1, groups=1, bias=True, padding_mode='zeros')
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1), # Output size: 32 x H x W
            nn.BatchNorm2d(32),
            # ReLU(inplace=False)
            nn.ReLU(),
            # MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
            nn.MaxPool2d(kernel_size=2, stride=2)  # Output size: 32 x H/2 x W/2
        )

        # Second convolutional layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1), # Output size: 64 x H/2 x W/2
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Output size: 64 x H/4 x W/4
        )

        # Third convolutional layer
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1), # Output size: 128 x H/4 x W/4
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Output size: 128 x H/8 x W/8
        )

        # Flatten layer
        self.flatten = nn.Flatten()

        # Fully connected layers
        # Linear(in_features, out_features, bias=True)
        self.fc1 = nn.Linear(128 * (H // 8) * (W // 8), 256)  # Adjust based on the input image size
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_class)  # Adjust the output size for the number of classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        x = self.flatten(x)  # Flatten the output for the fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x) # using nn.CrossEntropyLoss() as your loss func, then you do not need to specify softmax, it is already inside the loss function.

        return x

# Define the CIFAR10Data class using torchvision's CIFAR-10
class CIFAR10Data(Dataset):
    def __init__(self, root="./data", train=True, transform=None, download=True):
        # Initialize the CIFAR-10 dataset from torchvision
        self.cifar10 = datasets.CIFAR10(root=root, train=train, transform=transform, download=download)
        
    def __len__(self):
        # Return the number of samples
        return len(self.cifar10)

    def __getitem__(self, index):
        # Get the image and target label for the specified index
        image, target = self.cifar10[index]
        
        # Convert the target to a PyTorch tensor (if needed, though CIFAR-10 should already be in tensor format)
        target = torch.tensor(target, dtype=torch.long)
        
        return image, target        

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# Use GPU if available, otherwise stick with cpu
use_cuda = torch.cuda.is_available()
torch.manual_seed(123)
device = torch.device("cuda" if use_cuda else "cpu")
print(device)

# Hyperparameters
output_size = 10  # For NIST36 (36 classes)
#learning_rate = 2e-3
learning_rate = 1e-3
max_iters = 100
batch_size = 32

# Initialize datasets and dataloaders
# Load the full training dataset
# Define transformations (normalization based on CIFAR-10 mean and std)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])
full_train_dataset = CIFAR10Data(root="../data", train=True, transform=transform, download=True)
test_dataset = CIFAR10Data(root="../data", train=False, transform=transform, download=True)

# Split the full training dataset into train and validation sets
train_size = int(0.8 * len(full_train_dataset))  # 80% for training
valid_size = len(full_train_dataset) - train_size  # 20% for validation
train_dataset, valid_dataset = random_split(full_train_dataset, [train_size, valid_size])

# Create DataLoaders for train, validation, and test sets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, loss, and optimizer
input_size_w = np.array(train_dataset[0][0]).shape[-2]
input_size_h = np.array(train_dataset[0][0]).shape[-1]
model = Net(output_size, input_size_w, input_size_h).to(device)
model.apply(init_weights)
criterion = nn.CrossEntropyLoss()  # Loss function
#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)        
optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))

# Define a learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

# Training loop
train_losses = []
valid_losses = []
train_accuracies = []
valid_accuracies = []
for itr in range(max_iters):
    ############################## Training ###########################
    model.train()  # Set the model to training mode
    total_train_loss = 0
    total_train_correct = 0    
    
    for xb, yb in train_loader:
        optimizer.zero_grad()  # Zero the parameter gradients

        # Forward pass
        xb = xb.reshape(-1, 3, 32, 32)
        y_hat = model(xb)
        
        # Compute loss
        loss = criterion(y_hat, yb)
        total_train_loss += loss.item()        
        
        # Compute accuracy
        _, predicted = torch.max(y_hat, 1)
        total_train_correct += (predicted == yb).float().sum().item()
        
        # Backward pass and update
        loss.backward()
        optimizer.step()
    
    avg_train_loss = total_train_loss / len(train_loader)
    avg_train_acc = total_train_correct / len(train_dataset)
    train_losses.append(avg_train_loss)
    train_accuracies.append(avg_train_acc)

    ############################## Validation ###########################
    model.eval()  # Set the model to evaluation mode
    total_valid_loss = 0
    total_valid_correct = 0
    
    with torch.no_grad():  # Disable gradient computation for validation
        for xb, yb in valid_loader:
            # Forward pass
            xb = xb.reshape(-1, 3, 32, 32)
            y_hat = model(xb)
            
            # Compute loss and accuracy
            loss = criterion(y_hat, yb)
            total_valid_loss += loss.item()
            
            _, predicted = torch.max(y_hat, 1)
            total_valid_correct += (predicted == yb).float().sum().item()
    
    # Calculate average validation loss and accuracy
    avg_valid_loss = total_valid_loss / len(valid_loader)
    avg_valid_acc = total_valid_correct / len(valid_dataset)
    valid_losses.append(avg_valid_loss)
    valid_accuracies.append(avg_valid_acc)

    # Step the scheduler at the end of each epoch
    scheduler.step()

    # Print current learning rate for tracking
    current_lr = scheduler.get_last_lr()[0]
    
    print(f"Epoch {itr+1}/{max_iters}, "
          f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}, "
          f"Valid Loss: {avg_valid_loss:.4f}, Valid Acc: {avg_valid_acc:.4f}, "
          f"Learning rate after epoch {itr+1}: {current_lr:.6f}")

############################## Validation ###########################
model.eval()  # Set the model to evaluation mode
total_test_correct = 0
    
with torch.no_grad():  # Disable gradient computation for validation
    for xb, yb in test_loader:
        # Forward pass
        xb = xb.reshape(-1, 3, 32, 32)
        y_hat = model(xb)
            
        # Compute loss and accuracy
        loss = criterion(y_hat, yb)
            
        _, predicted = torch.max(y_hat, 1)
        total_test_correct += (predicted == yb).float().sum().item()

test_accuracy = total_test_correct/len(test_dataset)
print(f"Test Acc: {test_accuracy:.4f}")

# Plotting loss and accuracy over time
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(valid_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Time')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(valid_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy Over Time')
plt.legend()

plt.show()
print("Training complete.")