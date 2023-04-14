# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Create Fully Connected Network
# Set up class
class NN(nn.Module):
    def __init__(self, input_size, hidden_layer_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, num_classes)

    def forward(self, x): # take in x
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Test out class
model = NN(784, 50, 10)
x = torch.randn(64, 784) # minibatch size
print(model(x).shape) # Good! output is right size, 10 output

# Set Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Hyperparameters
input_size = 784 # Going to work with MNIST which is 28x28 pixel img
num_classes = 10 # number of output digits
learning_rate = .001 # slow it down, Murphy Brown
batch_size = 64
num_epochs = 1

# Load Data
# MNIST
train_dataset = datasets.MNIST(root = 'dataset/', train = True, transform = transforms.ToTensor(), download = True)
# Now create training loader
train_loader = DataLoader(dataset = train_dataset, shuffle= True, batch_size=batch_size)
# Test now
test_dataset = datasets.MNIST(root = 'dataset/', train = False, transform = transforms.ToTensor(), download = True)
# Now create training loader
test_loader = DataLoader(dataset=test_dataset, shuffle= True, batch_size = batch_size)

# Initialize Network
model = NN(input_size=input_size, hidden_layer_size=90, num_classes=num_classes).to(device) # Create model and pass to device

# Loss and Optimizer
# Going to use cross-entropy loss (discrete output)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate) # Use Adam because it's decent enough
# Train Network
# Loop over epochs
for epoch in range(num_epochs):
    # Go through batches
    for batch_idx, (data, targets) in enumerate(train_loader): # gives you data, targets

        # Pass data to device
        data = data.to(device=device)
        # Same for targets
        targets = targets.to(device=device)

        # Reshape images into one vector
        data = data.reshape(data.shape[0], -1) # Flattens into single dimension

        # Forward pass
        scores = model(data)
        loss = criterion(scores, targets)

        # Backprop
        optimizer.zero_grad() # Don't forget!
        loss.backward()

        # G-Descent
        optimizer.step() # Update weights

# Check Accuracy
def check_acc(loader, model):
    """Pass loader and model, conduct test"""

    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")

    num_corr = 0
    num_samples = 0

    model.eval() # Set model to evaluation mode

    with torch.no_grad(): # Don't worry about gradients
        for x, y in loader:
            x = x.to(device = device)
            y = y.to(device = device)
            x = x.reshape(x.shape[0],-1)#Unroll

            scores = model(x) # Predict
            _, predictions = scores.max(1)
            num_corr += (predictions==y).sum()
            num_samples += predictions.size(0)

        # Compute accuracy
    print(f'Got {num_corr} / {num_samples} with acc {(float(num_corr)/float(num_samples))*100:.2f}')

    # Return model to train mode
    model.train()

check_acc(train_loader, model)
check_acc(test_loader,model)
