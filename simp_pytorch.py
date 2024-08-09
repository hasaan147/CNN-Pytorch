import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image

# Define the CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Define ReLU and other layers
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Compute flattened size
        self._to_linear = None
        self.convs = nn.Sequential(
            self.conv1,
            self.relu,
            self.pool,
            self.conv2,
            self.relu,
            self.pool
        )
        self._get_linear_size()
        
        self.fc1 = nn.Linear(self._to_linear, 128)  # Adjusted size
        self.fc2 = nn.Linear(128, 10)

    def _get_linear_size(self):
        # Pass a dummy tensor through conv layers to compute the output size
        x = torch.randn(1, 1, 28, 28)  # Dummy input tensor
        x = self.convs(x)
        self._to_linear = x.numel()  # Number of features after flattening

    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Hyperparameters
batch_size = 64
learning_rate = 0.001
num_epochs = 5

# Load the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

@st.cache(allow_output_mutation=True)
def get_dataloaders():
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

train_loader, test_loader = get_dataloaders()

# Initialize model, loss function, and optimizer
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Streamlit app
st.title('CNN Training and Evaluation with PyTorch')

# Training
if st.button('Train Model'):
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        st.write(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    st.success('Training complete!')

# Testing
if st.button('Evaluate Model'):
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        st.write(f'Accuracy of the network on the 10000 test images: {accuracy:.2f}%')

# Upload and Predict
st.subheader('Upload Image for Prediction')
uploaded_file = st.file_uploader("Choose an image...", type="png")

if uploaded_file is not None:
    # Load and preprocess image
    image = Image.open(uploaded_file).convert('L')
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Predict
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        st.write(f'Predicted digit: {predicted.item()}')
