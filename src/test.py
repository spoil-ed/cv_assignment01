import os
import pickle
import logging
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np

# Define the CNN model
class Conv3LayerNN(nn.Module):
    def __init__(
        self, input_shape=(32, 32, 3), num_classes=10,
        conv1_filters=32, kernel1_size=3, conv1_stride=1, conv1_padding=1,
        conv2_filters=32, kernel2_size=3, conv2_stride=1, conv2_padding=0
    ):
        super(Conv3LayerNN, self).__init__()
        self.input_shape = input_shape
        H_in, W_in, C_in = input_shape

        # Conv1 layer
        self.conv1 = nn.Conv2d(C_in, conv1_filters, kernel1_size, stride=conv1_stride, padding=conv1_padding)
        self.bn1 = nn.BatchNorm2d(conv1_filters)
        self.pool1 = nn.MaxPool2d(2, stride=2)

        # Compute output shape after conv1 and pool1
        conv1_H_out = (H_in + 2 * conv1_padding - kernel1_size) // conv1_stride + 1
        conv1_W_out = (W_in + 2 * conv1_padding - kernel1_size) // conv1_stride + 1
        pool1_H_out = conv1_H_out // 2
        pool1_W_out = conv1_W_out // 2

        # Conv2 layer
        self.conv2 = nn.Conv2d(conv1_filters, conv2_filters, kernel2_size, stride=conv2_stride, padding=conv2_padding)
        self.bn2 = nn.BatchNorm2d(conv2_filters)
        self.pool2 = nn.MaxPool2d(2, stride=2)

        # Compute output shape after conv2 and pool2
        conv2_H_out = (pool1_H_out + 2 * conv2_padding - kernel2_size) // conv2_stride + 1
        conv2_W_out = (pool1_W_out + 2 * conv2_padding - kernel2_size) // conv2_stride + 1
        pool2_H_out = conv2_H_out // 2
        pool2_W_out = conv2_W_out // 2

        # Fully connected layer
        fc_input_size = pool2_H_out * pool2_W_out * conv2_filters
        self.fc = nn.Linear(fc_input_size, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

# Load CIFAR-10 data
def load_cifar10_batch(file):
    with open(file, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    data = batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) / 255.0  # [N, H, W, C]
    data = data.transpose(0, 3, 1, 2)  # 转换为 [N, C, H, W]
    labels = batch[b'labels']
    return torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

def load_cifar10_data(data_dir, test=False):
    train_data, train_labels = [], []
    for i in range(1, 6):
        X, y = load_cifar10_batch(os.path.join(data_dir, f'data_batch_{i}'))
        train_data.append(X)
        train_labels.append(y)
    train_data = torch.cat(train_data)
    train_labels = torch.cat(train_labels)
    test_data, test_labels = load_cifar10_batch(os.path.join(data_dir, 'test_batch'))

    num_samples = train_data.shape[0]
    perm = torch.randperm(num_samples)
    train_data_shuffled = train_data[perm]
    train_labels_shuffled = train_labels[perm]

    if test:
        val_size = 500
        valid_data, valid_labels = train_data_shuffled[-val_size:], train_labels_shuffled[-val_size:]
        train_size = 1000
        train_data, train_labels = train_data_shuffled[:train_size], train_labels_shuffled[:train_size]
    else:
        val_size = 5000
        valid_data, valid_labels = train_data_shuffled[-val_size:], train_labels_shuffled[-val_size:]
        train_data, train_labels = train_data_shuffled[:-val_size], train_labels_shuffled[:-val_size]

    return train_data, train_labels, valid_data, valid_labels, test_data, test_labels

# Setup logging
def setup_logging(log_dir="/home/spoil/cv/assignment01/experiments/logs"):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file)]
    )

# Training function
def train(model, train_data, train_labels, valid_data, valid_labels, lr, reg_lambda, batch_size, epochs, lr_decay=0.95, device='cuda'):
    model = model.to(device)
    train_dataset = TensorDataset(train_data, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataset = TensorDataset(valid_data, valid_labels)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=reg_lambda)
    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses, val_accs = [], [], []
    best_val_acc = 0.0

    for epoch in tqdm(range(epochs), desc="Training Progress"):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in valid_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                _, predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        val_loss /= len(valid_loader.dataset)
        val_acc = correct / total

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        logging.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        tqdm.write(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), '/home/spoil/cv/assignment01/experiments/best_model.pth')

        # Learning rate decay
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay

    # Load best model
    model.load_state_dict(torch.load('/home/spoil/cv/assignment01/experiments/best_model.pth'))
    return train_losses, val_losses, val_accs

# Hyperparameter search
def hyperparameter_search(lrs=[0.1, 0.01, 0.001], regs=[0.1, 0.01, 0.001], device='cuda'):
    logging.info("Starting hyperparameter search...")
    best_acc = 0.0
    best_lr, best_reg = None, None
    for lr in lrs:
        for reg in regs:
            logging.info(f"\n Hyperparameters: lr={lr}, reg={reg}")
            model = Conv3LayerNN()
            train_losses, val_losses, val_accs = train(
                model, train_data, train_labels, valid_data, valid_labels,
                lr=lr, reg_lambda=reg, batch_size=32, epochs=5, lr_decay=0.95, device=device
            )
            max_val_acc = max(val_accs)
            if max_val_acc >= best_acc:
                best_acc = max_val_acc
                best_lr = lr
                best_reg = reg
    return best_lr, best_reg

# Evaluation function
def evaluate(model, data, labels, batch_size=32, device='cuda'):
    model = model.to(device)
    dataset = TensorDataset(data, labels)
    loader = DataLoader(dataset, batch_size=batch_size)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in tqdm(loader, desc="Evaluation Progress"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    acc = correct / total
    print(f'Test Accuracy: {acc:.4f}')
    logging.info(f'Test Accuracy: {acc:.4f}')
    return acc

if __name__ == '__main__':
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Setup logging
    setup_logging(log_dir="/home/spoil/cv/assignment01/experiments/logs")
    logging.info("Starting CIFAR-10 training...")

    # Load data
    print("Loading CIFAR-10 data...")
    data_dir = "/home/spoil/cv/assignment01/data/cifar-10-python/cifar-10-batches-py/"
    train_data, train_labels, valid_data, valid_labels, test_data, test_labels = load_cifar10_data(data_dir, test=False)

    # Hyperparameter search
    print("Hyperparameter search...")
    best_lr, best_reg = hyperparameter_search(device=device)

    # Train model with best hyperparameters
    print("Training model...")
    model = Conv3LayerNN(
        input_shape=(32, 32, 3), num_classes=10,
        conv1_filters=32, kernel1_size=3, conv1_stride=1, conv1_padding=1,
        conv2_filters=32, kernel2_size=3, conv2_stride=1, conv2_padding=0
    )
    train(model, train_data, train_labels, valid_data, valid_labels, lr=best_lr, reg_lambda=best_reg, batch_size=32, epochs=20, lr_decay=0.95, device=device, visualize=True)

    # Evaluate on test set
    print("Evaluating on test set...")
    test_acc = evaluate(model, test_data, test_labels, batch_size=32, device=device)

    logging.info("Training completed.")