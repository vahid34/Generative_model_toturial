import torch
from torch import nn, optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Define the Autoencoder (AE) class
class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        # Encoder: Compresses the input into a lower-dimensional representation
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),  # Input layer (flattened 28x28 image) to 128 neurons
            nn.ReLU(),               # Activation function
            nn.Linear(128, 64),      # Hidden layer: 128 to 64 neurons
            nn.ReLU(), 
            nn.Linear(64, 36),       # Hidden layer: 64 to 36 neurons
            nn.ReLU(),
            nn.Linear(36, 18),       # Hidden layer: 36 to 18 neurons
            nn.ReLU(),
            nn.Linear(18, 9)         # Bottleneck layer: 18 to 9 neurons (compressed representation)
        )
        # Decoder: Reconstructs the input from the compressed representation
        self.decoder = nn.Sequential(
            nn.Linear(9, 18),        # Bottleneck layer to 18 neurons
            nn.ReLU(),
            nn.Linear(18, 36),       # Hidden layer: 18 to 36 neurons
            nn.ReLU(),
            nn.Linear(36, 64),       # Hidden layer: 36 to 64 neurons
            nn.ReLU(),
            nn.Linear(64, 128),      # Hidden layer: 64 to 128 neurons
            nn.ReLU(),
            nn.Linear(128, 28 * 28), # Output layer: 128 neurons to flattened 28x28 image
            nn.Sigmoid()             # Activation function to normalize output between 0 and 1
        )

    def forward(self, x):
        # Forward pass: Encode the input and then decode it
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Transform to convert images to tensors
tensor_transform = transforms.ToTensor()

# Load the MNIST dataset
dataset = datasets.MNIST(root="./data", train=True, download=True, transform=tensor_transform)

# Create a DataLoader to iterate through the dataset in batches
# - `batch_size=32`: Process 32 images at a time
# - `shuffle=True`: Shuffle the dataset at every epoch to improve training
loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=32, shuffle=True)

# Initialize the Autoencoder model
model = AE()

# Define the loss function
# - Mean Squared Error (MSE) is used to measure the difference between the input and reconstructed output
loss_function = nn.MSELoss()

# Define the optimizer
# - Adam optimizer is used for training the model
# - `lr=1e-3`: Learning rate
# - `weight_decay=1e-8`: L2 regularization to prevent overfitting
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)

# Number of training epochs
epochs = 20

# Lists to store outputs and losses for visualization
outputs = []
losses = []

# Check if a GPU is available and use it; otherwise, use the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # Move the model to the selected device

for epoch in range(epochs):  # Loop through each epoch
    for images, _ in loader:  # Iterate through the DataLoader to get batches of images
        # Flatten the images from (batch_size, 1, 28, 28) to (batch_size, 28*28) and move to the selected device
        images = images.view(-1, 28 * 28).to(device)
        
        # Forward pass: Pass the images through the autoencoder to get the reconstructed output
        reconstructed = model(images)
        
        # Compute the loss between the original images and the reconstructed images
        loss = loss_function(reconstructed, images)
        
        # Backpropagation: Reset gradients, compute gradients, and update weights
        optimizer.zero_grad()  # Clear previous gradients
        loss.backward()        # Compute gradients
        optimizer.step()       # Update model parameters
        
        # Store the loss value for visualization
        losses.append(loss.item())
    
    # Store the outputs for visualization after each epoch
    outputs.append((epoch, images, reconstructed))
    
    # Print the loss for the current epoch
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

# Plot the training loss over iterations
plt.style.use('fivethirtyeight')  # Set the plot style
plt.figure(figsize=(8, 5))        # Set the figure size
plt.plot(losses, label='Loss')    # Plot the loss values
plt.xlabel('Iterations')          # Label for the x-axis
plt.ylabel('Loss')                # Label for the y-axis
plt.legend()                      # Add a legend to the plot
plt.show()                        # Display the plot

model.eval()  # Set the model to evaluation mode (disables dropout, batch norm, etc.)
dataiter = iter(loader)  # Create an iterator for the DataLoader
images, _ = next(dataiter)  # Get the next batch of images from the DataLoader

# Flatten the images from (batch_size, 1, 28, 28) to (batch_size, 28*28) and move to the selected device
images = images.view(-1, 28 * 28).to(device)

# Pass the images through the autoencoder to get the reconstructed output
reconstructed = model(images)

# Plot the original and reconstructed images for comparison
fig, axes = plt.subplots(nrows=2, ncols=10, figsize=(10, 3))  # Create a 2x10 grid of subplots
for i in range(10):  # Loop through the first 10 images in the batch
    # Plot the original image in the first row
    axes[0, i].imshow(images[i].cpu().detach().numpy().reshape(28, 28), cmap='gray')
    axes[0, i].axis('off')  # Turn off the axis for better visualization
    
    # Plot the reconstructed image in the second row
    axes[1, i].imshow(reconstructed[i].cpu().detach().numpy().reshape(28, 28), cmap='gray')
    axes[1, i].axis('off')  # Turn off the axis for better visualization

plt.show()  # Display the plot

