import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.utils.data as data
import os

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

class SimpleCNNgray(nn.Module):
    def __init__(self, num_classes=6):
        super(SimpleCNNgray, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
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

def model_learning(input_model, data_path, model_export_name, transformations, epoch_number):

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = transformations

    # Create your dataset (replace 'your_data_dir' with the path to your dataset)
    dataset = datasets.ImageFolder(root=data_path, transform=data_transforms['train'])
    print(dataset.class_to_idx)
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
    model = input_model
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Check if a GPU is available and move the model to GPU if it is
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)

    # Training loop
    num_epochs = epoch_number
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
    torch.save(model.state_dict(), model_export_name)

    print('Finished Training')

if __name__ == '__main__':

    data_transforms_color = {
        'train': transforms.Compose([
            transforms.RandomRotation(180),  # Rotation for rotation invariance
            transforms.ToTensor(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.Normalize([0.4798, 0.4511, 0.4503], [0.1495, 0.1556, 0.1532])
        ])
    }

    data_transforms_gray = {
        'train': transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomRotation(180),  # Rotation for rotation invariance
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.Normalize([0.5049], [0.2935])
        ])
    }

    #model_learning(SimpleCNN(num_classes=6), os.path.join('D:','dicetoss_clean_test'), 'model_weights_color.pth', data_transforms_color, 50)
    model_learning(SimpleCNNgray(num_classes=6), os.path.join('D:','dicetoss_grayscale'), 'model_weights_gray.pth', data_transforms_gray, 20)