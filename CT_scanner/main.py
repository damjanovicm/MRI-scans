import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CustomDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = []
        self.labels = []
        for label_name in os.listdir(image_dir):
            label_path = os.path.join(image_dir, label_name)
            if os.path.isdir(label_path):
                for img_name in os.listdir(label_path):
                    if img_name.endswith(".jpg"):
                        self.images.append(os.path.join(label_path, img_name))
                        self.labels.append(label_name)
        self.label_map = {label: idx for idx, label in enumerate(sorted(set(self.labels)))}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.label_map[self.labels[idx]]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


def train_model(train_loader, model, criterion, optimizer, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 20 == 19:
                print(
                    f"Epoha {epoch + 1}/{num_epochs}, Batch {i + 1}/{len(train_loader)}, Gubitak: {running_loss / 20:.4f}")
                running_loss = 0.0
        print(f"Epoha {epoch + 1}/{num_epochs}, Srednji gubitak: {running_loss / len(train_loader):.4f}")


def evaluate_model(validation_loader, model):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in validation_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return accuracy_score(all_labels, all_preds)


def display_image(image_path, title):
    img = Image.open(image_path).convert('RGB')
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()


def predict_image(model, image_path, transform):
    model.eval()
    img = Image.open(image_path).convert('RGB')
    img = transform(img)
    img = img.unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        max_prob, pred_class = torch.max(probabilities, 1)
    return pred_class.item(), max_prob.item(), probabilities[0]


def main():
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = CustomDataset(image_dir='scans/Training', transform=data_transforms)
    dataset_size = len(dataset)
    train_size = int(0.9 * dataset_size)  # Veličina uzorka
    val_size = dataset_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_classes = len(dataset.label_map)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Početak treniranja modela...")
    train_model(train_loader, model, criterion, optimizer, num_epochs=1)

    print("Evaluacija modela...")
    accuracy = evaluate_model(val_loader, model)
    print(f"Tačnost modela: {accuracy * 100:.2f}%")

    test_image_path = 'scans/Testing/Gliom/Te-gl_0034.jpg'  # Promjeniti sliku za testiranje
    pred_class, max_prob, probabilities = predict_image(model, test_image_path, data_transforms)
    class_names = list(dataset.label_map.keys())

    print(f"Predikcija: {class_names[pred_class]}")
    print(f"Verovatnoća: {max_prob * 100:.2f}%")

    print("\nVerovatnoće po klasama:")
    for idx, prob in enumerate(probabilities):
        print(f"{class_names[idx]}: {prob.item() * 100:.2f}%")

    display_image(test_image_path, f"Predikcija: {class_names[pred_class]}")


if __name__ == '__main__':
    main()
