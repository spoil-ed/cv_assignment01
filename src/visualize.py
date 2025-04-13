import os
import math
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from tqdm import tqdm
from model import Conv3LayerNN
from utils import load_cifar10_data, setup_logging, get_project_paths

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

def plot_training_curves(train_losses, val_losses, val_accs, epochs, save_dir="experiments/plots"):
    """
    绘制训练集和验证集损失曲线，以及验证集准确率曲线
    """
    print("正在生成训练集和验证集损失曲线...")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss', color='blue', linewidth=2)
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss', color='orange', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    all_losses = np.concatenate([train_losses, val_losses])
    y_min, y_max = np.min(all_losses), np.max(all_losses)
    y_margin = (y_max - y_min) * 0.1
    plt.ylim(y_min - y_margin, y_max + y_margin)
    plt.savefig(os.path.join(save_dir, 'loss_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"损失曲线已保存至 {save_dir}/loss_curves.png")

    # 验证集准确率曲线
    print("正在生成验证集准确率曲线...")
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), val_accs, label='Validation Accuracy', color='green', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.grid(True)

    # 自适应 y 轴范围，添加 10% 边距，限制最小值为 0
    y_min, y_max = np.min(val_accs), np.max(val_accs)
    y_margin = (y_max - y_min) * 0.1 if y_max > y_min else 0.1  # 防止零范围
    plt.ylim(max(0, y_min - y_margin), y_max + y_margin)
    plt.savefig(os.path.join(save_dir, 'val_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"验证集准确率曲线已保存至 {save_dir}/val_accuracy.png")

def visualize_conv_kernels(model, save_dir="experiments/plots"):
    """
    可视化卷积核。
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 第一层卷积核
    print("正在生成第一层卷积核可视化...")
    W1 = cp.asnumpy(model.W1)
    num_kernels1 = min(32, W1.shape[0])  # 最多显示 32 个卷积核
    rows1 = math.ceil(math.sqrt(num_kernels1))
    cols1 = math.ceil(num_kernels1 / rows1)
    plt.figure(figsize=(10, 5))
    for i in range(min(32, W1.shape[0])):
        kernel = W1[i]
        kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min() + 1e-8)
        plt.subplot(rows1, cols1, i + 1)
        plt.imshow(kernel)
        plt.axis('off')
    plt.suptitle('First Layer Convolution Kernels')
    plt.savefig(os.path.join(save_dir, 'conv1_kernels.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("第一层卷积核已保存至 {}/conv1_kernels.png".format(save_dir))

    # 第二层卷积核
    print("正在生成第二层卷积核可视化...")
    W2 = cp.asnumpy(model.W2)
    num_kernels2 = min(32, W2.shape[0])  # 最多显示 32 个卷积核
    rows2 = math.ceil(math.sqrt(num_kernels2))
    cols2 = math.ceil(num_kernels2 / rows2)
    plt.figure(figsize=(10, 5))
    for i in range(min(32, W2.shape[0])):
        kernel = W2[i, :, :, 0]  
        kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min() + 1e-8)
        plt.subplot(rows2, cols2, i + 1)
        plt.imshow(kernel, cmap='gray')
        plt.axis('off')
    plt.suptitle('Second Layer Convolution Kernels (First Channel)')
    plt.savefig(os.path.join(save_dir, 'conv2_kernels.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("第二层卷积核已保存至 {}/conv2_kernels.png".format(save_dir))

def plot_fc_weights(model, save_dir="experiments/plots"):
    """
    绘制全连接层权重（W3）的热图。
    """
    print("正在生成全连接层权重热图...")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 全连接权重
    W3 = cp.asnumpy(model.W3)  # 形状 (512, 10) 或类似
    plt.figure(figsize=(10, 6))
    sns.heatmap(W3, cmap='RdBu', center=0, cbar=True)
    plt.xlabel('Class')
    plt.ylabel('Feature')
    plt.title('Fully Connected Layer Weights')
    plt.savefig(os.path.join(save_dir, 'fc_weights.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("全连接层权重热图已保存至 {}/fc_weights.png".format(save_dir))

def plot_param_distribution(model, save_dir="experiments/plots"):
    """
    绘制权重参数分布直方图和各偏置参数（b1, b2, b3, beta1, beta2）按通道的偏置值柱状图。
    """
    print("正在生成参数分布图...")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    weights = [
        cp.asnumpy(model.W1).ravel(),
        cp.asnumpy(model.W2).ravel(),
        cp.asnumpy(model.W3).ravel()
    ]
    weights = np.concatenate(weights)
    biases = {
        'b1': cp.asnumpy(model.b1),
        'b2': cp.asnumpy(model.b2),
        'b3': cp.asnumpy(model.b3),
        'beta1': cp.asnumpy(model.beta1),
        'beta2': cp.asnumpy(model.beta2)
    }

    # 绘制权重分布
    plt.figure(figsize=(6, 5))
    plt.hist(weights, bins=50, color='blue', alpha=0.7)
    plt.xlim([-1, 1])  
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    plt.title('Weight Parameter Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'weight_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"权重分布直方图已保存至 {save_dir}/weight_distribution.png")

    # 为每个偏置参数绘制按通道的偏置值柱状图
    for bias_name, bias_values in biases.items():
        if bias_values.ndim > 1:
            bias_values = bias_values.mean(axis=tuple(range(1, bias_values.ndim)))
        else:
            bias_values = bias_values

        num_channels = bias_values.shape[0]
        channels = np.arange(1, num_channels + 1)
        plt.figure(figsize=(10, 5))
        plt.bar(channels, bias_values, color='orange', alpha=0.7)
        plt.xlabel('Channel Index')
        plt.ylabel(f'{bias_name} Value')
        plt.title(f'{bias_name} Value per Channel')
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{bias_name}_value_per_channel.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"{bias_name} 按通道偏置值柱状图已保存至 {save_dir}/{bias_name}_value_per_channel.png")

def plot_bn_params(model, gamma1_history, beta1_history, gamma2_history, beta2_history, epochs, save_dir="experiments/plots"):
    """
    绘制批归一化参数（gamma1, gamma2, beta1, beta2）随轮次的变化趋势。
    """
    print("正在生成批归一化参数趋势图...")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 计算每轮次参数的均值
    gamma1_vals = [cp.asnumpy(g).mean() for g in gamma1_history]
    gamma2_vals = [cp.asnumpy(g).mean() for g in gamma2_history]
    beta1_vals = [cp.asnumpy(b).mean() for b in beta1_history]
    beta2_vals = [cp.asnumpy(b).mean() for b in beta2_history]

    # 绘制 gamma 参数趋势
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), gamma1_vals, label='Gamma1', color='blue', linewidth=2)
    plt.plot(range(1, epochs + 1), gamma2_vals, label='Gamma2', color='orange', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Gamma Value')
    plt.title('Batch Normalization Gamma Parameters')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'bn_gamma.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("批归一化 Gamma 参数趋势图已保存至 {}/bn_gamma.png".format(save_dir))

    # 绘制 beta 参数趋势
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), beta1_vals, label='Beta1', color='blue', linewidth=2)
    plt.plot(range(1, epochs + 1), beta2_vals, label='Beta2', color='orange', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Beta Value')
    plt.title('Batch Normalization Beta Parameters')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'bn_beta.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("批归一化 Beta 参数趋势图已保存至 {}/bn_beta.png".format(save_dir))

def plot_class_accuracy(model, data_dir="data/cifar-10-batches-py", save_dir="experiments/plots", batch_size=32):
    """
    绘制每个类别的预测正确率柱状图。
    """
    print("正在加载 CIFAR-10 测试数据以计算类别正确率...")
    try:
        _, _, _, _, test_data, test_labels = load_cifar10_data(data_dir, test=False)
        print("测试数据加载成功！")
    except Exception as e:
        logging.error("加载测试数据失败：{}".format(e))
        print(f"错误：加载测试数据失败：{e}")
        return

    print("正在对测试集进行预测以计算类别正确率...")
    num_samples = test_data.shape[0]
    all_preds = []
    for i in tqdm(range(0, num_samples, batch_size),desc = "处理预测进度"):
        batch_data = test_data[i:i + batch_size]
        probs = model.forward(batch_data)
        preds = np.argmax(cp.asnumpy(probs), axis=1)
        all_preds.append(preds)
        cp.get_default_memory_pool().free_all_blocks()
    all_preds = np.concatenate(all_preds)
    test_labels_np = cp.asnumpy(test_labels)
    print("预测完成！")

    # 计算每个类别的正确率
    print("正在计算每个类别的正确率...")
    num_classes = len(CIFAR10_CLASSES)
    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)
    for pred, label in zip(all_preds, test_labels_np):
        class_total[label] += 1
        if pred == label:
            class_correct[label] += 1
    class_accuracy = class_correct / (class_total + 1e-8)  # 避免除零

    # 绘制柱状图
    print("正在生成类别正确率柱状图...")
    plt.figure(figsize=(12, 6))
    plt.bar(CIFAR10_CLASSES, class_accuracy, color='skyblue', edgecolor='black')
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.title('Class-wise Prediction Accuracy')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'class_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("类别正确率柱状图已保存至 {}/class_accuracy.png".format(save_dir))

def plot_hyperparam_tuning_lr_reg(save_dir="experiments/plots"):
    """
    绘制学习率和正则化系数的调参热图。
    """
    paths = get_project_paths()
    if save_dir is None:
        save_dir = paths['plots_dir']

    print("正在加载 lr 和 reg 调参实验结果...")
    results_path = paths['hyperparam_lr_reg_path']
    try:
        data = np.load(results_path)
        lrs = data['lrs']
        regs = data['regs']
        val_accs = data['val_accs'] 
        print("lr 和 reg 调参实验结果加载成功！")
    except FileNotFoundError:
        logging.warning(f"未找到调参结果文件 {results_path}，将使用示例数据...")
        print(f"未找到调参结果文件 {results_path}，使用示例数据...")
        lrs = [0.001, 0.01, 0.1]
        regs = [0.0, 0.0001, 0.001]
        val_accs = np.random.rand(len(lrs), len(regs)) * 0.4 + 0.5

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("正在生成 lr 和 reg 调参热图...")
    plt.figure(figsize=(8, 6))
    sns.heatmap(val_accs, xticklabels=[f"{r:.4f}" for r in regs], yticklabels=[f"{lr:.3f}" for lr in lrs],
                annot=True, fmt=".3f", cmap="YlGnBu", cbar=True)
    plt.xlabel('Regularization Coefficient')
    plt.ylabel('Learning Rate')
    plt.title('Validation Accuracy (lr vs reg)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'hyperparam_lr_reg.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"lr 和 reg 调参热图已保存至 {save_dir}/hyperparam_lr_reg.png")

def plot_hyperparam_tuning_conv_filters(save_dir=None):
    """
    绘制第一层和第二层卷积核个数的调参热图。
    """
    paths = get_project_paths()
    if save_dir is None:
        save_dir = paths['plots_dir']

    print("正在加载卷积核个数调参实验结果...")
    results_path = paths['hyperparam_conv_filters_path']
    try:
        data = np.load(results_path)
        conv1_filters = data['conv1_filters']
        conv2_filters = data['conv2_filters']
        val_accs = data['val_accs']
        print("卷积核个数调参实验结果加载成功！")
    except FileNotFoundError:
        logging.warning(f"未找到调参结果文件 {results_path}，将使用示例数据...")
        print(f"未找到调参结果文件 {results_path}，使用示例数据...")
        conv1_filters = [16, 32, 64]
        conv2_filters = [32, 64, 128]
        val_accs = np.random.rand(len(conv1_filters), len(conv2_filters)) * 0.4 + 0.5

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("正在生成卷积核个数调参热图...")
    plt.figure(figsize=(8, 6))
    sns.heatmap(val_accs, xticklabels=conv2_filters, yticklabels=conv1_filters,
                annot=True, fmt=".3f", cmap="YlGnBu", cbar=True)
    plt.xlabel('Conv2 Filters')
    plt.ylabel('Conv1 Filters')
    plt.title('Validation Accuracy (Conv Filters)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'hyperparam_conv_filters.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"卷积核个数调参热图已保存至 {save_dir}/hyperparam_conv_filters.png")


def plot_hyperparam_tuning_conv_filters_equal(save_dir=None):
    """
    绘制第一层和第二层卷积核个数相等的调参折线图。
    """
    paths = get_project_paths()
    if save_dir is None:
        save_dir = paths['plots_dir']

    print("正在加载相等卷积核个数调参实验结果...")
    results_path = paths['hyperparam_conv_filters_equal_path']
    try:
        data = np.load(results_path)
        conv_filters = data['conv_filters']
        val_accs = data['val_accs']  # 形状 (n_filters,)
        print("相等卷积核个数调参实验结果加载成功！")
    except FileNotFoundError:
        logging.warning(f"未找到调参结果文件 {results_path}，将使用示例数据...")
        print(f"未找到调参结果文件 {results_path}，使用示例数据...")
        conv_filters = np.array([32, 64, 128, 256])
        val_accs = np.random.rand(len(conv_filters)) * 0.4 + 0.5

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("正在生成相等卷积核个数调参折线图...")
    plt.figure(figsize=(8, 6))
    plt.plot(conv_filters, val_accs, marker='o', linestyle='-', color='b', linewidth=2, markersize=8)
    for x, y in zip(conv_filters, val_accs):
        plt.text(x, y + 0.005, f"{y:.3f}", ha='center', va='bottom', fontsize=10)
    plt.xlabel('Number of Filters (Conv1 = Conv2)')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy vs Equal Conv Filters')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'hyperparam_conv_filters_equal.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"相等卷积核个数调参折线图已保存至 {save_dir}/hyperparam_conv_filters_equal.png")


if __name__ == '__main__':
    paths = get_project_paths()
    
    # 输入自定义保存路径
    print("开始运行可视化程序...")
    save_dir = paths['plots_dir']
    print(f"图像将保存至：{save_dir}")

    # 设置日志
    setup_logging(log_dir=os.path.join(os.path.dirname(save_dir), "logs"))
    logging.info("开始生成可视化结果...")

    # 加载训练数据
    print("正在加载训练数据...")
    try:
        data = np.load(paths['data_npz_path'])
        train_losses = data['train_losses']
        val_losses = data['val_losses']
        val_accs = data['val_accs']
        epochs = len(train_losses)
        print("训练数据加载成功！")
    except FileNotFoundError:
        logging.warning("未找到训练数据文件，将使用示例数据...")
        print("未找到训练数据文件，使用随机示例数据...")
        epochs = 20
        train_losses = np.random.rand(epochs) * 2
        val_losses = np.random.rand(epochs) * 2
        val_accs = np.random.rand(epochs) * 0.5 + 0.5

    # 加载模型
    print("正在初始化模型...")
    model = Conv3LayerNN(
        input_shape=(32, 32, 3), num_classes=10,
        conv1_filters=32, kernel1_size=(3, 3), conv1_stride=1, conv1_padding=1,
        conv2_filters=64, kernel2_size=(3, 3), conv2_stride=1, conv2_padding=0
    )
    print("模型初始化完成！")

    # 加载最佳权重
    print("正在加载模型权重...")
    try:
        weights = np.load(paths['weights_path'])
        model.W1 = cp.asarray(weights['W1'])
        model.b1 = cp.asarray(weights['b1'])
        model.W2 = cp.asarray(weights['W2'])
        model.b2 = cp.asarray(weights['b2'])
        model.W3 = cp.asarray(weights['W3'])
        model.b3 = cp.asarray(weights['b3'])
        model.gamma1 = cp.asarray(weights['gamma1'])
        model.beta1 = cp.asarray(weights['beta1'])
        model.gamma2 = cp.asarray(weights['gamma2'])
        model.beta2 = cp.asarray(weights['beta2'])
        weights.close()
        print("模型权重加载成功！")
    except FileNotFoundError:
        logging.warning("未找到权重文件，将使用随机初始化的权重...")
        print("未找到权重文件，使用随机初始化的权重...")

    # 模拟批归一化参数历史记录（实际需从训练中记录）
    print("正在生成模拟的批归一化参数数据...")
    gamma1_history = [model.gamma1 + cp.random.randn(*model.gamma1.shape) * 0.1 for _ in range(epochs)]
    gamma2_history = [model.gamma2 + cp.random.randn(*model.gamma2.shape) * 0.1 for _ in range(epochs)]
    beta1_history = [model.beta1 + cp.random.randn(*model.beta1.shape) * 0.1 for _ in range(epochs)]
    beta2_history = [model.beta2 + cp.random.randn(*model.beta2.shape) * 0.1 for _ in range(epochs)]
    print("批归一化参数数据准备完成！")

    print("请输入 CIFAR-10 数据目录（用于类别正确率计算，默认：data/cifar-10-batches-py）：")
    data_dir = input().strip()
    if not data_dir:
        data_dir = paths['data_dir']

    # 可视化
    plot_training_curves(train_losses, val_losses, val_accs, epochs, save_dir)
    visualize_conv_kernels(model, save_dir)
    plot_fc_weights(model, save_dir)
    plot_param_distribution(model, save_dir)
    plot_bn_params(model, gamma1_history, beta1_history, gamma2_history, beta2_history, epochs, save_dir)
    plot_class_accuracy(model, data_dir, save_dir)
    plot_hyperparam_tuning_lr_reg(save_dir)
    plot_hyperparam_tuning_conv_filters(save_dir)
    print("所有可视化生成完成！")
    logging.info(f"可视化生成完成，所有图像已保存至 {save_dir}。")