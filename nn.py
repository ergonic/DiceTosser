import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.utils.data as data

# Define a simple CNN architecture
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 32 * 32, 512)
        self.fc1_5 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 32 * 32)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc1_5(x))
        #x = self.dropout(x)
        x = self.fc2(x)
        return x

def model_learning():

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(180),  # Rotation for rotation invariance
            transforms.ToTensor(),
            transforms.ColorJitter(0.2,0.2,0.2,0.1),
            transforms.Normalize([0.4798, 0.4511, 0.4503], [0.1495, 0.1556, 0.1532])
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4798, 0.4511, 0.4503], [0.1495, 0.1556, 0.1532])
        ]),
    }

    # Create your dataset (replace 'your_data_dir' with the path to your dataset)
    dataset = datasets.ImageFolder(root='D:\\dicetoss_clean_test', transform=data_transforms['train'])

    # Note: Further code would involve setting up DataLoaders for the dataset,
    # creating an instance of the SimpleCNN, defining a loss function and optimizer,
    # and writing the training loop.

    # This code provides a basic structure. Adapt it to your specific dataset and requirements.

    # Assuming 'dataset' is already created with ImageFolder as shown in the previous snippet

    # Splitting the dataset into train and validation sets
    train_size = int(0.8 * len(dataset))
    validation_size = len(dataset) - train_size
    train_dataset, validation_dataset = data.random_split(dataset, [train_size, validation_size])

    # Creating data loaders for the train and validation sets
    train_loader = data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    validation_loader = data.DataLoader(validation_dataset, batch_size=64, shuffle=True, num_workers=0)

    # Create an instance of the SimpleCNN and define loss function and optimizer
    model = SimpleCNN(num_classes=6)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Check if a GPU is available and move the model to GPU if it is
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)

    # Training loop
    num_epochs = 200
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Print statistics
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

        # Validation
        model.eval()  # Set the model to evaluation mode
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in validation_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Validation Accuracy: {100 * correct / total}%')

    # Save the model's state_dict
    torch.save(model.state_dict(), 'model_weights_color.pth')

    print('Finished Training')

    '''
    # Create the model instance
    model = SimpleCNN(num_classes=6)
    
    # Load the saved state dict
    model.load_state_dict(torch.load('model_weights.pth'))
    
    # Set the model to evaluation mode
    model.eval()
    '''

if __name__ == '__main__':
    model_learning()