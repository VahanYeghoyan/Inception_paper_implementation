import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from Inception import *



data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


def create_data_loaders(root_folder, batch_size=64, shuffle=True, num_workers=2):
    dataset = datasets.ImageFolder(root=root_folder, transform=data_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader


model = Inception()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train_model(model, train_loader, valid_loader, num_epochs=3):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0


        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {epoch_loss:.4f}')


        model.eval()
        total_correct = 0
        total_top5_correct = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, labels in tqdm(valid_loader):
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)

                total_samples += labels.size(0)
                total_correct += (predicted == labels).sum().item()

                _, top5_predicted = torch.topk(outputs, 5, dim=1)
                total_top5_correct += sum([1 for i in range(labels.size(0)) if labels[i] in top5_predicted[i]])

        accuracy = total_correct / total_samples
        top5_accuracy = total_top5_correct / total_samples

        print(f'Validation Accuracy after epoch {epoch+1}: {accuracy:.4f}')
        print(f'Validation Top-5 Accuracy after epoch {epoch+1}: {top5_accuracy:.4f}')

    print('Training complete.')



train_loader = create_data_loaders("C:\\Users\\vahan.yeghoyan\\Desktop\\train_images")
valid_loader = create_data_loaders("C:\\Users\\vahan.yeghoyan\\Desktop\\train_images")


train_model(model, train_loader, valid_loader)
