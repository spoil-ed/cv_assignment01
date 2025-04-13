import os
import csv
import pickle
import logging
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.interpolate import griddata

# 项目路径
def get_project_paths():
    # 获取项目根目录（假设 utils.py 所在目录的上一级为根目录）
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    return {
        'project_root': project_root,
        'data_dir': os.path.join(project_root, 'data', 'cifar-10-batches-py'),
        'experiments_dir': os.path.join(project_root, 'experiments'),
        'logs_dir': os.path.join(project_root, 'experiments', 'logs'),
        'plots_dir': os.path.join(project_root, 'experiments', 'plots'),
        'weights_path': os.path.join(project_root, 'experiments', 'results', 'best_model_weights.npz'),
        'hyperparam_lr_reg_path': os.path.join(project_root, 'experiments', 'results', 'hyperparam_lr_reg.npz'),
        'hyperparam_conv_filters_path': os.path.join(project_root, 'experiments', 'results', 'hyperparam_conv_filters.npz'),
        'hyperparam_conv_filters_equal_path': os.path.join(project_root, 'experiments', 'results', 'hyperparam_conv_filters_equal.npz'),
        'data_npz_path': os.path.join(project_root, 'experiments', 'data.npz'),
    }

# 批次处理
def load_cifar10_batch(file):
    with open(file, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    
    data = batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) / 255.0
    data = cp.asarray(data)
    labels = cp.asarray(batch[b'labels'])
    return data, labels

# 总处理
def load_cifar10_data(data_dir, test=False):
    train_data, train_labels = [], []

    for i in range(1, 6):
        X, y = load_cifar10_batch(os.path.join(data_dir, f'data_batch_{i}'))
        train_data.append(X)
        train_labels.append(y)
    train_data = cp.concatenate(train_data)
    train_labels = cp.concatenate(train_labels)
    test_data, test_labels = load_cifar10_batch(os.path.join(data_dir, 'test_batch'))

    num_samples = train_data.shape[0]
    perm = cp.random.permutation(num_samples)
    train_data_shuffled = train_data[perm]
    train_labels_shuffled = train_labels[perm]

    
    if test:
        val_size = 50
        valid_data, valid_labels = train_data_shuffled[-val_size:], train_labels_shuffled[-val_size:]
        train_size = 500
        train_data, train_labels = train_data_shuffled[:train_size], train_labels_shuffled[:train_size]
    else:
        val_size = 500
        valid_data, valid_labels = train_data_shuffled[-val_size:], train_labels_shuffled[-val_size:]
        train_data, train_labels = train_data_shuffled[:-val_size], train_labels_shuffled[:-val_size]

    return train_data, train_labels, valid_data, valid_labels, test_data, test_labels

# 日志
def setup_logging(log_dir=None):
    paths = get_project_paths()
    if log_dir is None:
        log_dir = paths['logs_dir']
    
    # 创建日志目录
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 使用当前时间生成日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
        ]
    )