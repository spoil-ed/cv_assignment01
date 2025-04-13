import logging
import os
import cupy as cp
import numpy as np
from tqdm import tqdm
from datetime import datetime
from model import Conv3LayerNN
from utils import load_cifar10_data, setup_logging, get_project_paths
import matplotlib.pyplot as plt

# 训练
def train(model, train_data, train_labels, valid_data, valid_labels, lr, reg_lambda, batch_size, epochs, lr_decay=0.95, visualize=False):
    paths = get_project_paths()
    num_samples = train_data.shape[0]
    best_val_acc = 0.0
    train_losses, train_accs, val_losses, val_accs = [], [], [], []
    
    for epoch in tqdm(range(epochs), desc="Training Progress"):
        # 使用 CuPy 打乱数据
        perm = cp.random.permutation(num_samples)
        train_data_shuffled = train_data[perm]
        train_labels_shuffled = train_labels[perm]
        
        for i in tqdm(range(0, num_samples, batch_size), total=num_samples // batch_size, desc="Inner Training Progress"):
            X_batch = train_data_shuffled[i:i + batch_size]
            y_batch = train_labels_shuffled[i:i + batch_size]
            
            probs = model.forward(X_batch)
            loss = model.compute_loss(probs, y_batch, reg_lambda)
            grads = model.backward(X_batch, y_batch, probs, reg_lambda)
            model.update_params(grads, lr, reg_lambda, max_grad_norm=5.0)
        
        # 评估训练集（部分数据）将 CuPy 数组转回 NumPy 以计算准确率（因为 np.argmax 不支持 CuPy 直接操作）
        train_probs = model.forward(train_data[:1000])
        train_loss = model.compute_loss(train_probs, train_labels[:1000], reg_lambda)
        train_acc = model.accuracy(cp.asnumpy(train_probs), cp.asnumpy(train_labels[:1000]))

        # 评估验证集（批处理）
        val_batch_size = 32
        val_probs = []
        for i in tqdm(range(0, valid_data.shape[0], val_batch_size), desc="Validation Progress"):
            valid_data_batch = valid_data[i:i + val_batch_size]
            val_probs.append(cp.asnumpy(model.forward(valid_data_batch)))
            cp.get_default_memory_pool().free_all_blocks()
        val_probs = np.concatenate(val_probs)
        val_acc = model.accuracy(val_probs, cp.asnumpy(valid_labels))
        val_loss = model.compute_loss(cp.asarray(val_probs), valid_labels, reg_lambda)

        train_losses.append(float(cp.asnumpy(train_loss)))  # 转换为 Python 标量
        train_accs.append(train_acc)
        val_losses.append(float(cp.asnumpy(val_loss)))
        val_accs.append(val_acc)
        
        
        # 记录 epoch 结果
        logging.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # 将 CuPy 数组转回 NumPy 以保存
            W1_np = cp.asnumpy(model.W1)
            b1_np = cp.asnumpy(model.b1)
            W2_np = cp.asnumpy(model.W2)
            b2_np = cp.asnumpy(model.b2)
            W3_np = cp.asnumpy(model.W3)
            b3_np = cp.asnumpy(model.b3)
            gamma1_np = cp.asnumpy(model.gamma1)
            beta1_np = cp.asnumpy(model.beta1)
            gamma2_np = cp.asnumpy(model.gamma2)
            beta2_np = cp.asnumpy(model.beta2)
            np.savez(paths['weights_path'],
                     W1=W1_np, b1=b1_np, W2=W2_np, b2=b2_np, W3=W3_np, b3=b3_np, gamma1 = gamma1_np, beta1 = beta1_np, gamma2 = gamma2_np, beta2 = beta2_np)
        
        # lr *= lr_decay
        if epoch % 5 == 0:
            lr *= 0.5
        tqdm.write(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        cp.get_default_memory_pool().free_all_blocks()
    
    if visualize:
        plt.plot(range(epochs), train_losses, label='Train Loss')
        plt.plot(range(epochs), val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.show()
        plt.plot(range(epochs), val_accs, label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Validation Accuracy')
        plt.legend()
        plt.show()
    np.savez(paths['data_npz_path'], train_losses=train_losses, train_accs = train_accs, val_losses=val_losses, val_accs=val_accs)
    return train_losses, train_accs, val_losses, val_accs

def hyperparameter_search(ModelClass = Conv3LayerNN, 
                          lrs = [0.001, 0.01, 0.1],               # 学习率
                          regs = [0.0001, 0.001, 0.01],            # 正则化系数
                          conv1_filters_list = [16, 32, 64],
                          conv2_filters_list = [16, 32, 64],
                          batch_size=32, 
                          epochs=5, 
                          lr_decay=0.95,
                          save_dir = None
                          ):
    logging.info("Starting hyperparameter search...")
    if save_dir is None:
        save_dir = paths['experiments_dir']

    best_acc = 0.0
    best_lr = None
    best_reg = None
    best_conv1_filters = None
    best_conv2_filters = None

    
    print("开始实验 1: 学习率和正则化系数...")
    lrs = [0.001, 0.01, 0.1]               # 学习率
    regs = [0.0001, 0.001, 0.01]            # 正则化系数
    fixed_conv1_filters = 32                # 固定第一层卷积核个数
    fixed_conv2_filters = 64                # 固定第二层卷积核个数
    val_accs_lr_reg = np.zeros((len(lrs), len(regs)))

    
    for i, lr in enumerate(lrs):
        for j, reg in enumerate(regs):
            print(f"实验: lr={lr}, reg={reg}, conv1_filters={fixed_conv1_filters}, conv2_filters={fixed_conv2_filters}")
            model = Conv3LayerNN(
                input_shape=(32, 32, 3), num_classes=10,
                conv1_filters=fixed_conv1_filters, kernel1_size=(3, 3), conv1_stride=1, conv1_padding=1,
                conv2_filters=fixed_conv2_filters, kernel2_size=(3, 3), conv2_stride=1, conv2_padding=0
            )
            train_losses, train_accs, val_losses, val_accs= train(model, train_data, train_labels, valid_data, 
                                                                  valid_labels,epochs=epochs, batch_size=32, lr=lr, reg_lambda=reg)
            val_acc = np.max(val_accs)
            if val_acc >= best_acc:
                best_lr = lr
                best_reg = reg
                best_acc = np.max(val_accs)
            val_accs_lr_reg[i, j] = val_acc
            logging.info(f"实验完成：lr={lr}, reg={reg}, val_acc={val_acc:.4f}")

    # 保存 lr 和 reg 实验结果
    print("正在保存 lr 和 reg 调参结果...")
    np.savez(os.path.join(save_dir, "hyperparam_lr_reg.npz"),
             lrs=np.array(lrs),
             regs=np.array(regs),
             val_accs=val_accs_lr_reg)
    print(f"lr 和 reg 调参结果已保存至 {save_dir}/hyperparam_lr_reg.npz")

    
    best_acc = 0.0
    best_conv1_filters = None
    best_conv2_filters = None
    print("开始实验 2：卷积核个数...")
    fixed_lr = best_lr                      
    fixed_reg = best_reg                    
    val_accs_conv_filters = np.zeros((len(conv1_filters_list), len(conv2_filters_list)))

    for i, conv1_filters in enumerate(conv1_filters_list):
        for j, conv2_filters in enumerate(conv2_filters_list):
            print(f"实验：conv1_filters={conv1_filters}, conv2_filters={conv2_filters}, lr={fixed_lr}, reg={fixed_reg}")
            model = Conv3LayerNN(
                input_shape=(32, 32, 3), num_classes=10,
                conv1_filters=conv1_filters, kernel1_size=(3, 3), conv1_stride=1, conv1_padding=1,
                conv2_filters=conv2_filters, kernel2_size=(3, 3), conv2_stride=1, conv2_padding=0
            )
            _, _, _, val_accs= train(model, train_data, train_labels, valid_data, 
                                                                  valid_labels,epochs=epochs, batch_size=32, lr=fixed_lr, reg_lambda=fixed_reg, lr_decay=lr_decay)
            val_acc = np.max(val_accs)
            if val_acc >= best_acc:
                best_conv1_filters = conv1_filters
                best_conv2_filters = conv2_filters
                best_acc = val_acc
            val_accs_conv_filters[i, j] = val_acc
            logging.info(f"实验完成：conv1_filters={conv1_filters}, conv2_filters={conv2_filters}, "
                         f"val_acc={val_acc:.4f}")

    # 保存 conv_filters 实验结果
    print("正在保存卷积核个数调参结果...")
    np.savez(os.path.join(save_dir, "hyperparam_conv_filters.npz"),
             conv1_filters=np.array(conv1_filters_list),
             conv2_filters=np.array(conv2_filters_list),
             val_accs=val_accs_conv_filters)
    print(f"卷积核个数调参结果已保存至 {save_dir}/hyperparam_conv_filters.npz")
    

    return best_lr, best_reg, best_conv1_filters, best_conv2_filters
            
def evaluate(model, data, labels, batch_size = 32):
    probs = []
    for i in tqdm(range(0, data.shape[0], batch_size), desc="Evaluation Progress"):
        data_batch = data[i:i + batch_size]
        probs.append(cp.asnumpy(model.forward(data_batch)))
        cp.get_default_memory_pool().free_all_blocks()
    probs = np.concatenate(probs)
    acc = model.accuracy(probs, cp.asnumpy(labels))
    print(f'Test Accuracy: {acc:.4f}')
    logging.info(f'Test Accuracy: {acc:.4f}')
    return acc

if __name__ == '__main__':
    paths = get_project_paths()
    # 设置日志
    setup_logging()
    logging.info("Starting CIFAR-10 training...")

    print("Loading CIFAR-10 data...")
    train_data, train_labels, valid_data, valid_labels, test_data, test_labels = load_cifar10_data(data_dir=paths['data_dir'], test = True)
    
    print("hyperparameter search...")
    best_lr, best_reg, best_conv1_filters, best_conv2_filters = hyperparameter_search(ModelClass = Conv3LayerNN, 
                                                                                    lrs = [0.015, 0.01, 0.095], 
                                                                                    regs = [0.1, 0.01, 0.001],
                                                                                    conv1_filters_list = [16, 32, 64],
                                                                                    conv2_filters_list = [16, 32, 64],
                                                                                    lr_decay=0.95,
                                                                                    batch_size=32, 
                                                                                    epochs=5, 
                                                                                    )
                                                                                        
    print("Training model...")

    model = Conv3LayerNN(
                input_shape=(32, 32, 3), num_classes=10,
                conv1_filters=best_conv1_filters, kernel1_size=(3, 3), conv1_stride=1, conv1_padding=1,
                conv2_filters=best_conv2_filters, kernel2_size=(3, 3), conv2_stride=1, conv2_padding=0
                )
    train_losses, train_accs, val_losses, val_accs = train(model, train_data, train_labels, valid_data, valid_labels, 
                                                          lr=best_lr, reg_lambda=best_reg, batch_size=32, epochs=20, lr_decay=0.995, visualize=False)

    print("Evaluating on test set...")
    batch_size = 32  
    test_acc = evaluate(model, test_data, test_labels, batch_size)

    logging.info("Training completed.")