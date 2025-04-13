import numpy as np
from model_numpy import Conv3LayerNN
import pickle
import os
from tqdm import tqdm
import logging
from datetime import datetime

def load_cifar10_batch(file):
    with open(file, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    data = batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) / 255.0  # [N, 32, 32, 3]
    labels = np.array(batch[b'labels'])
    return data, labels

def load_cifar10_data(data_dir, test = False):
    train_data, train_labels = [], []

    for i in range(1, 6):
        X, y = load_cifar10_batch(os.path.join(data_dir, f'data_batch_{i}'))
        # train_data.append(X[:size])
        # train_labels.append(y[:size])
        train_data.append(X)
        train_labels.append(y)
    train_data = np.concatenate(train_data)
    train_labels = np.concatenate(train_labels)
    test_data, test_labels = load_cifar10_batch(os.path.join(data_dir, 'test_batch'))
    return train_data, train_labels, test_data, test_labels

def setup_logging(log_dir="/home/spoil/cv/assignment01/experiments/logs"):
    # 创建日志目录
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 使用当前时间生成日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,  # 设置日志级别（INFO、DEBUG、WARNING 等）
        format="%(asctime)s [%(levelname)s] %(message)s",  # 日志格式
        handlers=[
            logging.FileHandler(log_file),  # 保存到文件
            logging.StreamHandler()  # 输出到控制台
        ]
    )

def train(model, X_train, y_train, X_val, y_val, lr, reg_lambda, batch_size, epochs, lr_decay=0.95):
    num_samples = X_train.shape[0]
    best_val_acc = 0.0
    train_losses, val_losses, val_accs = [], [], []
    
    for epoch in tqdm(range(epochs), desc="Training Progress"):
        perm = np.random.permutation(num_samples)
        X_train_shuffled = X_train[perm]
        y_train_shuffled = y_train[perm]
        
        for i in tqdm(range(0, num_samples, batch_size), total = num_samples // batch_size,desc="Inner Training Progress"):
            X_batch = X_train_shuffled[i:i + batch_size]
            y_batch = y_train_shuffled[i:i + batch_size]
            
            probs = model.forward(X_batch)
            loss = model.compute_loss(probs, y_batch, reg_lambda)
            grads = model.backward(X_batch, y_batch, probs, reg_lambda)
            model.update_params(grads, lr, reg_lambda, max_grad_norm=5.0)
        
        train_probs = model.forward(X_train[:1000]) 
        # train_probs = model.forward(X_train[:1000])  # 用部分数据评估
        train_loss = model.compute_loss(train_probs, y_train[:1000], reg_lambda)
        # train_loss = model.compute_loss(train_probs, y_train[:1000], reg_lambda)
        train_acc = np.mean(np.argmax(train_probs, axis=1) == y_train[:1000])
        # train_acc = np.mean(np.argmax(train_probs, axis=1) == y_train[:1000])
        val_probs = model.forward(X_val)
        # print(np.argmax(val_probs, axis=1))
        val_loss = model.compute_loss(val_probs, y_val, reg_lambda)
        val_acc = np.mean(np.argmax(val_probs, axis=1) == y_val)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # 记录 epoch 结果
        logging.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            np.savez('/home/spoil/cv/assignment01/experiments/best_model_weights.npz', W1=model.W1, b1=model.b1, 
                     W2=model.W2, b2=model.b2, W3=model.W3, b3=model.b3)
        
        lr *= lr_decay
        tqdm.write(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    return train_losses, val_losses, val_accs



if __name__ == '__main__':
    # 设置日志
    setup_logging(log_dir="/home/spoil/cv/assignment01/experiments/logs")
    logging.info("Starting CIFAR-10 training...")

    print("Loading CIFAR-10 data...")
    data_dir = "/home/spoil/cv/assignment01/data/cifar-10-python/cifar-10-batches-py/"
    X_train, y_train, X_test, y_test = load_cifar10_data(data_dir)
    
    num_samples = X_train.shape[0]
    perm = np.random.permutation(num_samples)
    X_train_shuffled = X_train[perm]
    y_train_shuffled = y_train[perm]

    # train_size = 1000
    val_size = 500
    # val_size = 5000
    X_val, y_val = X_train[-val_size:], y_train[-val_size:]
    X_train, y_train = X_train[:-val_size], y_train[:-val_size]
    # X_train, y_train = X_train[:train_size], y_train[:train_size]
    
    print("Training model...")
    model = Conv3LayerNN()
    train_losses, val_losses, val_accs = train(model, X_train, y_train, X_val, y_val, 
                                               lr=0.1, reg_lambda=0.001, batch_size=128, epochs=20, lr_decay=0.995)

    print("Evaluating on test set...")
    test_probs = model.forward(X_test)
    test_acc = np.mean(np.argmax(test_probs, axis=1) == y_test)
    print(f'Test Accuracy: {test_acc:.4f}')
    logging.info(f'Test Accuracy: {test_acc:.4f}')

    logging.info("Training completed.")
