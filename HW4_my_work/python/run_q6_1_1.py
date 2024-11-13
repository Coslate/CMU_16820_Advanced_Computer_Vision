
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import scipy.io
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

# Define the network architecture
class FullyConnectedNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FullyConnectedNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        h1_out = self.sigmoid(self.layer1(x))
        y_hat = self.output_layer(h1_out)
        #y_hat = self.softmax(self.output_layer(h1_out)) #Don't need softmax if using nn.CrossEntropyLoss() as the loss function.
        return y_hat

### Sample dataloader For Q.6.1 ####
class NIST36_Data(torch.utils.data.Dataset):
    def __init__(self, type):
        self.type = type
        self.data = scipy.io.loadmat(f"../data/nist36_{type}.mat")
        self.inputs, self.one_hot_target = (
            self.data[f"{self.type}_data"],
            self.data[f"{self.type}_labels"],
        )
        self.target = np.argmax(self.one_hot_target, axis=1)

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, index):
        inputs = torch.from_numpy(self.inputs[index]).type(torch.float32)
        target = torch.tensor(self.target[index]).type(torch.LongTensor)
        return inputs, target

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# Hyperparameters
hidden_size = 256
output_size = 36  # For NIST36 (36 classes)
#learning_rate = 2e-3
learning_rate = 1e-3
max_iters = 100
batch_size = 32

# Initialize datasets and dataloaders
train_data = NIST36_Data(type="train")
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_data = NIST36_Data(type="valid")
valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
test_data = NIST36_Data(type="test")
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Initialize model, loss, and optimizer
input_size = np.array(train_data[0][0]).shape[0]
model = FullyConnectedNetwork(input_size, hidden_size, output_size)
model.apply(init_weights)
criterion = nn.CrossEntropyLoss()  # Loss function
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Define a learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

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
    avg_train_acc = total_train_correct / len(train_data)
    train_losses.append(avg_train_loss)
    train_accuracies.append(avg_train_acc)

    ############################## Validation ###########################
    model.eval()  # Set the model to evaluation mode
    total_valid_loss = 0
    total_valid_correct = 0
    
    with torch.no_grad():  # Disable gradient computation for validation
        for xb, yb in valid_loader:
            # Forward pass
            y_hat = model(xb)
            
            # Compute loss and accuracy
            loss = criterion(y_hat, yb)
            total_valid_loss += loss.item()
            
            _, predicted = torch.max(y_hat, 1)
            total_valid_correct += (predicted == yb).float().sum().item()
    
    # Calculate average validation loss and accuracy
    avg_valid_loss = total_valid_loss / len(valid_loader)
    avg_valid_acc = total_valid_correct / len(valid_data)
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
        y_hat = model(xb)
            
        # Compute loss and accuracy
        loss = criterion(y_hat, yb)
            
        _, predicted = torch.max(y_hat, 1)
        total_test_correct += (predicted == yb).float().sum().item()

test_accuracy = total_test_correct/len(test_data)        
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