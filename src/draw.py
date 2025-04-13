import logging
import cupy as cp
import numpy as np
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import os
import gc

# 假设 model.py 和 utils.py 存在
from model import Conv3LayerNN
from utils import load_cifar10_data, setup_logging

# 训练函数（保持不变）
def train(
    model,
    train_data,
    train_labels,
    valid_data,
    valid_labels,
    lr,
    reg_lambda,
    batch_size,
    epochs,
    lr_decay=0.95,
    max_grad_norm=5.0,
    val_batch_size=32,
    visualize=False,
):
    num_samples = train_data.shape[0]
    best_val_acc = 0.0
    train_losses, train_accs, val_losses, val_accs = [], [], [], []

    for epoch in tqdm(range(epochs), desc="Training Progress"):
        perm = cp.random.permutation(num_samples)
        train_data_shuffled = train_data[perm]
        train_labels_shuffled = train_labels[perm]

        for i in tqdm(
            range(0, num_samples, batch_size),
            total=num_samples // batch_size,
            desc="Inner Training Progress",
        ):
            X_batch = train_data_shuffled[i:i + batch_size]
            y_batch = train_labels_shuffled[i:i + batch_size]

            probs = model.forward(X_batch)
            loss = model.compute_loss(probs, y_batch, reg_lambda)
            grads = model.backward(X_batch, y_batch, probs, reg_lambda)
            model.update_params(grads, lr, reg_lambda, max_grad_norm=max_grad_norm)

        train_probs = model.forward(train_data[:1000])
        train_loss = model.compute_loss(train_probs, train_labels[:1000], reg_lambda)
        train_acc = model.accuracy(cp.asnumpy(train_probs), cp.asnumpy(train_labels[:1000]))

        val_probs = []
        for i in tqdm(range(0, valid_data.shape[0], val_batch_size), desc="Validation Progress"):
            valid_data_batch = valid_data[i:i + val_batch_size]
            val_probs.append(cp.asnumpy(model.forward(valid_data_batch)))
            cp.get_default_memory_pool().free_all_blocks()
        val_probs = np.concatenate(val_probs)
        val_acc = model.accuracy(val_probs, cp.asnumpy(valid_labels))
        val_loss = model.compute_loss(cp.asarray(val_probs), valid_labels, reg_lambda)

        train_losses.append(float(cp.asnumpy(train_loss)))
        train_accs.append(train_acc)
        val_losses.append(float(cp.asnumpy(val_loss)))
        val_accs.append(val_acc)

        logging.info(
            f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            W1_np = cp.asnumpy(model.W1)
            b1_np = cp.asnumpy(model.b1)
            W2_np = cp.asnumpy(model.W2)
            b2_np = cp.asnumpy(model.b2)
            W3_np = cp.asnumpy(model.W3)
            b3_np = cp.asnumpy(model.b3)
            np.savez(
                "/home/spoil/cv/assignment01/experiments/best_model_weights.npz",
                W1=W1_np,
                b1=b1_np,
                W2=W2_np,
                b2=b2_np,
                W3=W3_np,
                b3=b3_np,
            )

        lr *= lr_decay
        tqdm.write(
            f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        cp.get_default_memory_pool().free_all_blocks()

    weights = np.load("/home/spoil/cv/assignment01/experiments/best_model_weights.npz")
    model.W1 = cp.asarray(weights["W1"])
    model.b1 = cp.asarray(weights["b1"])
    model.W2 = cp.asarray(weights["W2"])
    model.b2 = cp.asarray(weights["b2"])
    model.W3 = cp.asarray(weights["W3"])
    model.b3 = cp.asarray(weights["b3"])
    weights.close()

    if visualize:
        plt.plot(range(epochs), train_losses, label="Train Loss")
        plt.plot(range(epochs), val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.show()
        plt.plot(range(epochs), val_accs, label="Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Validation Accuracy")
        plt.legend()
        plt.show()

    return train_losses, train_accs, val_losses, val_accs

# 绘图函数
def plot_hyperparameter_results(results, save_dir="/home/spoil/cv/assignment01/experiments/plots"):
    """
    为每个超参数绘制验证准确率折线图。

    参数：
        results: 试验结果列表，每个元素为包含超参数和 max_val_acc 的字典
        save_dir: 保存图像的目录
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    param_groups = {}
    for result in results:
        param_name = result["param_tested"]
        if param_name not in param_groups:
            param_groups[param_name] = []
        param_groups[param_name].append(result)

    for param_name, group in param_groups.items():
        values = [result[param_name] for result in group]
        accs = [result["max_val_acc"] if not np.isnan(result["max_val_acc"]) else 0 for result in group]
        value_labels = [str(v) for v in values]

        plt.figure(figsize=(8, 6))
        plt.plot(range(len(values)), accs, marker="o", linestyle="-", label=param_name)
        plt.xticks(range(len(values)), value_labels, rotation=45)
        plt.xlabel(param_name)
        plt.ylabel("Validation Accuracy")
        plt.title(f"Validation Accuracy vs {param_name}")
        plt.legend()
        plt.grid(True)

        save_path = os.path.join(save_dir, f"{param_name}_acc_plot.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        logging.info(f"Saved plot for {param_name} to {save_path}")

# 修改后的超参数搜索函数
def hyperparameter_search(
    train_data,
    train_labels,
    valid_data,
    valid_labels,
    ModelClass=Conv3LayerNN,
    hyperparam_grid=None,
    default_params=None,
    csv_path="/home/spoil/cv/assignment01/experiments/hyperparam_results.csv",
    plot_dir="/home/spoil/cv/assignment01/experiments/plots",
):
    """
    依次调节超参数，优化验证集准确率，保存结果到 CSV 并绘制图像。

    参数：
        train_data, train_labels, valid_data, valid_labels: 训练和验证数据
        ModelClass: 模型类，默认为 Conv3LayerNN
        hyperparam_grid: 字典，键为超参数名，值为候选值列表
        default_params: 字典，初始默认超参数值
        csv_path: 保存结果的 CSV 文件路径
        plot_dir: 保存图像的目录

    返回：
        best_params: 最佳超参数组合
    """
    logging.info("Starting sequential hyperparameter search...")

    # 默认超参数网格（限制范围以避免内存问题）
    if hyperparam_grid is None:
        hyperparam_grid = {
            "lr": [0.01, 0.001, 0.0001],
            "reg_lambda": [0.01, 0.001, 0.0001],
            "batch_size": [16, 32, 64],
            "epochs": [5, 10],
            "lr_decay": [0.9, 0.95],
            "max_grad_norm": [1.0, 5.0],
            "val_batch_size": [16, 32],
            "conv1_filters": [16, 32],
            "kernel1_size": [(3, 3)],
            "conv1_stride": [1],
            "conv1_padding": [1],
            "conv2_filters": [16, 32],
            "kernel2_size": [(3, 3)],
            "conv2_stride": [1],
            "conv2_padding": [0],
            "pool_size": [2],
            "pool_stride": [2],
            "bn_epsilon": [1e-5, 1e-4],
        }

    # 默认超参数初始值
    if default_params is None:
        default_params = {
            "lr": 0.01,
            "reg_lambda": 0.01,
            "batch_size": 32,
            "epochs": 5,
            "lr_decay": 0.95,
            "max_grad_norm": 5.0,
            "val_batch_size": 32,
            "conv1_filters": 16,
            "kernel1_size": (3, 3),
            "conv1_stride": 1,
            "conv1_padding": 1,
            "conv2_filters": 16,
            "kernel2_size": (3, 3),
            "conv2_stride": 1,
            "conv2_padding": 0,
            "pool_size": 2,
            "pool_stride": 2,
            "bn_epsilon": 1e-5,
        }

    best_params = default_params.copy()
    best_acc = 0.0
    results = []

    # 确保使用正确的 CUDA 设备
    cp.cuda.Device(0).use()

    # 按顺序优化每个超参数
    for param_name in hyperparam_grid:
        if len(hyperparam_grid[param_name]) > 1:
            logging.info(f"\nOptimizing {param_name}...")
            param_best_acc = 0.0
            param_best_value = best_params[param_name]

            for value in hyperparam_grid[param_name]:
                current_params = best_params.copy()
                current_params[param_name] = value

                logging.info(f"Testing {param_name} = {value}")
                try:
                    # 验证超参数
                    H_in, W_in = 32, 32  # CIFAR-10 输入尺寸
                    kernel1_h, kernel1_w = (
                        current_params["kernel1_size"]
                        if isinstance(current_params["kernel1_size"], tuple)
                        else (current_params["kernel1_size"], current_params["kernel1_size"])
                    )
                    conv1_H_out = (H_in + 2 * current_params["conv1_padding"] - kernel1_h) // current_params["conv1_stride"] + 1
                    conv1_W_out = (W_in + 2 * current_params["conv1_padding"] - kernel1_w) // current_params["conv1_stride"] + 1
                    if conv1_H_out <= 0 or conv1_W_out <= 0:
                        raise ValueError(f"Invalid conv1 output size: {conv1_H_out}x{conv1_W_out}")

                    model = ModelClass(
                        input_shape=(32, 32, 3),
                        num_classes=10,
                        conv1_filters=current_params["conv1_filters"],
                        kernel1_size=current_params["kernel1_size"],
                        conv1_stride=current_params["conv1_stride"],
                        conv1_padding=current_params["conv1_padding"],
                        conv2_filters=current_params["conv2_filters"],
                        kernel2_size=current_params["kernel2_size"],
                        conv2_stride=current_params["conv2_stride"],
                        conv2_padding=current_params["conv2_padding"],
                        pool_size=current_params["pool_size"],
                        pool_stride=current_params["pool_stride"],
                        bn_epsilon=current_params["bn_epsilon"],
                    )
                    train_losses, train_accs, val_losses, val_accs = train(
                        model,
                        train_data,
                        train_labels,
                        valid_data,
                        valid_labels,
                        lr=current_params["lr"],
                        reg_lambda=current_params["reg_lambda"],
                        batch_size=current_params["batch_size"],
                        epochs=current_params["epochs"],
                        lr_decay=current_params["lr_decay"],
                        max_grad_norm=current_params["max_grad_norm"],
                        val_batch_size=current_params["val_batch_size"],
                    )
                    max_val_acc = np.max(val_accs)

                    result = current_params.copy()
                    result["max_val_acc"] = max_val_acc
                    result["param_tested"] = param_name
                    results.append(result)

                    if max_val_acc > param_best_acc:
                        param_best_acc = max_val_acc
                        param_best_value = value
                        if max_val_acc > best_acc:
                            best_acc = max_val_acc
                            best_params = current_params.copy()

                    logging.info(f"{param_name}={value}, Val Acc: {max_val_acc:.4f}")
                except Exception as e:
                    logging.warning(f"Failed for {param_name}={value}: {str(e)}")
                    result = current_params.copy()
                    result["max_val_acc"] = np.nan
                    result["param_tested"] = param_name
                    result["error"] = str(e)
                    results.append(result)
                    continue
                finally:
                    # 清理 GPU 内存
                    cp.get_default_memory_pool().free_all_blocks()
                    cp.get_default_pinned_memory_pool().free_all_blocks()
                    del model
                    gc.collect()

            best_params[param_name] = param_best_value
            logging.info(f"Best {param_name}: {param_best_value}, Best Val Acc: {param_best_acc:.4f}")

    # 保存结果到 CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(csv_path, index=False)
    logging.info(f"Results saved to {csv_path}")

    logging.info(f"Best hyperparameters: {best_params}, Best Val Acc: {best_acc:.4f}")
    return best_params

# 评估函数（保持不变）
def evaluate(model, data, labels, batch_size=32):
    probs = []
    for i in tqdm(range(0, data.shape[0], batch_size), desc="Evaluation Progress"):
        data_batch = data[i:i + batch_size]
        probs.append(cp.asnumpy(model.forward(data_batch)))
        cp.get_default_memory_pool().free_all_blocks()
    probs = np.concatenate(probs)
    acc = model.accuracy(probs, cp.asnumpy(labels))
    print(f"Test Accuracy: {acc:.4f}")
    logging.info(f"Test Accuracy: {acc:.4f}")
    return acc

# 主程序
if __name__ == "__main__":
    setup_logging(log_dir="/home/spoil/cv/assignment01/experiments/logs")
    logging.info("Starting CIFAR-10 training...")

    print("Loading CIFAR-10 data...")
    data_dir = "/home/spoil/cv/assignment01/data/cifar-10-python/cifar-10-batches-py/"
    train_data, train_labels, valid_data, valid_labels, test_data, test_labels = load_cifar10_data(
        data_dir, test=False
    )

    hyperparam_grid = {
            "lr": [0.015, 0.01, 0.0095],
            "reg_lambda": [0.005],
            "batch_size": [32],
            "epochs": [5],
            "lr_decay": [0.9, 0.95],
            "max_grad_norm": [5.0],
            "val_batch_size": [32],
            "conv1_filters": [32],
            "kernel1_size": [(3, 3)],
            "conv1_stride": [1],
            "conv1_padding": [1],
            "conv2_filters": [64],
            "kernel2_size": [(3, 3)],
            "conv2_stride": [1],
            "conv2_padding": [0],
            "pool_size": [2],
            "pool_stride": [2],
            "bn_epsilon": [1e-5],
        }
    default_params = {
            "lr": 0.05,
            "reg_lambda": 0.005,
            "batch_size": 32,
            "epochs": 5,
            "lr_decay": 0.95,
            "max_grad_norm": 5.0,
            "val_batch_size": 32,
            "conv1_filters": 32,
            "kernel1_size": (3, 3),
            "conv1_stride": 1,
            "conv1_padding": 1,
            "conv2_filters": 64,
            "kernel2_size": (3, 3),
            "conv2_stride": 1,
            "conv2_padding": 0,
            "pool_size": 2,
            "pool_stride": 2,
            "bn_epsilon": 1e-5,
    }

    print("Hyperparameter search...")
    best_params = hyperparameter_search(
        train_data,
        train_labels,
        valid_data,
        valid_labels,
        hyperparam_grid=hyperparam_grid,
        ModelClass=Conv3LayerNN,
        csv_path="/home/spoil/cv/assignment01/experiments/results/hyperparam_results.csv",
        plot_dir="/home/spoil/cv/assignment01/experiments/plots",
    )

    print("Training model with best hyperparameters...")
    try:
        # 清理 GPU 内存
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

        model = Conv3LayerNN(
            input_shape=(32, 32, 3),
            num_classes=10,
            conv1_filters=best_params["conv1_filters"],
            kernel1_size=best_params["kernel1_size"],
            conv1_stride=best_params["conv1_stride"],
            conv1_padding=best_params["conv1_padding"],
            conv2_filters=best_params["conv2_filters"],
            kernel2_size=best_params["kernel2_size"],
            conv2_stride=best_params["conv2_stride"],
            conv2_padding=best_params["conv2_padding"],
            pool_size=best_params["pool_size"],
            pool_stride=best_params["pool_stride"],
            bn_epsilon=best_params["bn_epsilon"],
        )
        train_losses, train_accs, val_losses, val_accs = train(
            model,
            train_data,
            train_labels,
            valid_data,
            valid_labels,
            lr=best_params["lr"],
            reg_lambda=best_params["reg_lambda"],
            batch_size=best_params["batch_size"],
            epochs=best_params["epochs"],
            lr_decay=best_params["lr_decay"],
            max_grad_norm=best_params["max_grad_norm"],
            val_batch_size=best_params["val_batch_size"],
            visualize=False,
        )

        print("Evaluating on test set...")
        test_acc = evaluate(model, test_data, test_labels, batch_size=best_params["val_batch_size"])

    except Exception as e:
        logging.error(f"Final training failed: {str(e)}")
        print(f"Final training failed: {str(e)}")
        raise

    logging.info("Training completed.")