
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F  # a lower level (compared to torch.nn) interface
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms, models
from torchvision.models import SqueezeNet1_1_Weights
import scipy.io
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

# Define the network architecture: SqueezeNet1_1
Net = models.squeezenet1_1(weights=SqueezeNet1_1_Weights)
# Replace the classifier layer to fit Flower17's 17 classes
num_classes = 17
Net.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
Net.num_classes = num_classes

# Freeze all layers except the classifier
for param in Net.features.parameters():
    param.requires_grad = False


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
output_size = 17  # For NIST36 (36 classes)
learning_rate = 2e-3
#learning_rate = 1e-5
max_iters = 100
batch_size = 32

# Initialize datasets and dataloaders
# Load the full training dataset
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_val_transform = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)

train_dataset = datasets.ImageFolder('../data/oxford-flowers17/train', transform=train_transform)
valid_dataset = datasets.ImageFolder('../data/oxford-flowers17/val', transform=test_val_transform)
test_dataset  = datasets.ImageFolder('../data/oxford-flowers17/test', transform=test_val_transform)

# Split the full training dataset into train and validation sets

# Create DataLoaders for train, validation, and test sets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, loss, and optimizer
input_size_c = np.array(train_dataset[0][0]).shape[-3]
input_size_w = np.array(train_dataset[0][0]).shape[-2]
input_size_h = np.array(train_dataset[0][0]).shape[-1]
model = Net
#model.apply(init_weights)
criterion = nn.CrossEntropyLoss()  # Loss function
#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)        
optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))

# Define a learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

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
        xb = xb.reshape(-1, input_size_c, input_size_w, input_size_h)
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
            xb = xb.reshape(-1, input_size_c, input_size_w, input_size_h)
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
        xb = xb.reshape(-1, input_size_c, input_size_w, input_size_h)
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