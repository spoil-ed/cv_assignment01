import os
import logging
import numpy as np
import cupy as cp
from PIL import Image
from model import Conv3LayerNN
from tqdm import tqdm
from utils import load_cifar10_data, setup_logging, get_project_paths

# CIFAR-10 类别名称
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

def preprocess_image(image_path):
    """
    预处理单张图像，转换为模型输入格式 (1, 32, 32, 3)。
    """
    print("正在预处理图像：{}...".format(image_path))
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((32, 32))  # 调整到 CIFAR-10 尺寸
        img_array = np.array(img) / 255.0  # 归一化到 [0, 1]
        img_array = img_array.reshape(1, 32, 32, 3)  # 形状为 (1, 32, 32, 3)
        img_array = cp.asarray(img_array)  # 转换为 CuPy 数组
        print("图像预处理完成！")
        return img_array
    except Exception as e:
        print(f"图像预处理失败：{e}")
        return None

def load_model(weights_path="experiments/best_model_weights.npz"):
    """
    加载模型和权重。
    """
    print("正在初始化模型...")
    model = Conv3LayerNN(
        input_shape=(32, 32, 3), num_classes=10,
        conv1_filters=32, kernel1_size=(3, 3), conv1_stride=1, conv1_padding=1,
        conv2_filters=64, kernel2_size=(3, 3), conv2_stride=1, conv2_padding=0
    )
    print("模型初始化完成！")

    print("正在加载模型权重：{}...".format(weights_path))
    try:
        weights = np.load(weights_path)
        model.W1 = cp.asarray(weights['W1'])
        model.b1 = cp.asarray(weights['b1'])
        model.W2 = cp.asarray(weights['W2'])
        model.b3 = cp.asarray(weights['b3'])
        model.W3 = cp.asarray(weights['W3'])
        model.b3 = cp.asarray(weights['b3'])
        model.gamma1 = cp.asarray(weights['gamma1'])
        model.beta1 = cp.asarray(weights['beta1'])
        model.gamma2 = cp.asarray(weights['gamma2'])
        model.beta2 = cp.asarray(weights['beta2'])
        weights.close()
        print("模型权重加载成功！")
    except FileNotFoundError:
        logging.error("未找到权重文件：{}".format(weights_path))
        print("错误：未找到权重文件：{}，无法进行预测！".format(weights_path))
        return None
    return model

def predict(model, data, labels=None, batch_size=32, is_single_image=False):
    if model is None:
        print("模型未加载，无法进行预测！")
        return

    # 批量预测（测试集）
    print("正在对测试集进行批量预测...")
    num_samples = data.shape[0]
    all_probs = []
    for i in tqdm(range(0, num_samples, batch_size), 
              total=(num_samples + batch_size - 1)//batch_size, 
              desc="处理批次"):
        batch_data = data[i:i + batch_size]
        batch_probs = model.forward(batch_data)
        all_probs.append(cp.asnumpy(batch_probs))
        cp.get_default_memory_pool().free_all_blocks()
    all_probs = np.concatenate(all_probs)
    print("批量预测完成！")

    # 计算准确率（如果提供了标签）
    if labels is not None:
        predictions = np.argmax(all_probs, axis=1)
        labels_np = cp.asnumpy(labels)
        accuracy = np.mean(predictions == labels_np)
        print(f"测试集准确率：{accuracy:.4f}")

    # 输出前几个样本的预测结果
    print("前 5 个样本的预测结果：")
    for i in range(min(5, num_samples)):
        pred_class = np.argmax(all_probs[i])
        print(f"样本 {i + 1}：预测类别 = {CIFAR10_CLASSES[pred_class]} (概率：{all_probs[i][pred_class]:.4f})")
    
    return all_probs

if __name__ == '__main__':
    paths = get_project_paths()
    
    # 设置日志
    print("开始运行预测程序...")
    setup_logging()
    logging.info("开始模型预测...")

    # 输入模型权重路径
    weights_path = input("请输入模型权重路径（默认：experiments/best_model_weights.npz）：").strip()
    if not weights_path:
        weights_path = paths['weights_path']

    # 加载模型
    model = load_model(weights_path)
    if model is None:
        exit(1)

    # 测试集预测
    data_dir = input("请输入 CIFAR-10 数据目录（默认：data/cifar-10-batches-py）：").strip()
    if not data_dir:
        data_dir = paths['data_dir']
    
    print("正在加载 CIFAR-10 测试数据...")
    try:
        _, _, _, _, test_data, test_labels = load_cifar10_data(data_dir, test=False)
        print("测试数据加载成功！")
    except Exception as e:
        logging.error("加载测试数据失败：{}".format(e))
        print(f"错误：加载测试数据失败：{e}")
        exit(1)
    
    # 进行预测
    predict(model, test_data, test_labels, batch_size=32, is_single_image=False)
    
    print("预测程序运行完成！")
    logging.info("预测完成。")