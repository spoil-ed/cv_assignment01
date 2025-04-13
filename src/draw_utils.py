import os
import csv
import pickle
import logging
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.interpolate import griddata
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为 SimHei
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def save_train(save_path, train_losses, train_acc, val_losses, val_accs, filename='train_metrics.csv'):
    # 目录存在
    os.makedirs(save_path, exist_ok=True)
    csv_file = os.path.join(save_path, filename)
    
    # 长度一致
    epochs = len(train_losses)
    if not (len(train_acc) == epochs == len(val_losses) == len(val_accs)):
        logging.error("Input lists must have the same length.")
        raise ValueError("train_losses, train_acc, val_losses, val_accs must have the same length")
    
    # 数据
    rows = [
        [i + 1, train_losses[i], train_acc[i], val_losses[i], val_accs[i]]
        for i in range(epochs)
    ]
    
    # 写入 CSV 文件
    with open(csv_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train_Loss', 'Train_Acc', 'Val_Loss', 'Val_Acc'])
        writer.writerows(rows)
    
def save_test(save_path, test_acc, filename='test_metrics.csv'):
    # 确保保存目录存在
    os.makedirs(save_path, exist_ok=True)
    csv_file = os.path.join(save_path, filename)
    
    # 写入 CSV 文件
    with open(csv_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Test_Accuracy'])
        writer.writerow([test_acc])
    
    logging.info(f"Test metrics saved to {csv_file}")

def plot_hyperparameter_final_acc(hyperparam_labels, final_accs):
    plt.figure(figsize=(12, 6))
    x = range(len(hyperparam_labels))
    plt.bar(x, final_accs, tick_label=hyperparam_labels, color='skyblue')
    plt.xlabel('超参数组合')
    plt.ylabel('第五轮验证准确率')
    plt.title('不同超参数组合的第五轮验证准确率')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# 绘制三维插值表面图并保存
def plot_3d_interpolation(lrs, regs, final_accs, save_path):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为 SimHei
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    # 将 lr 和 reg 转换为对数尺度
    log_lrs = np.log10(lrs)
    log_regs = np.log10(regs)
    
    # 创建原始数据点
    points = np.array([(log_lr, log_reg) for log_lr in log_lrs for log_reg in log_regs])
    values = np.array(final_accs)
    
    # 创建插值网格
    grid_x, grid_y = np.meshgrid(
        np.linspace(min(log_lrs), max(log_lrs), 100),
        np.linspace(min(log_regs), max(log_regs), 100)
    )
    
    # 进行插值（使用线性插值）
    grid_z = griddata(points, values, (grid_x, grid_y), method='linear')
    
    # 绘制三维表面图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制插值表面
    surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis', alpha=0.8)
    
    # 绘制原始数据点
    ax.scatter(log_lrs.repeat(len(log_regs)), np.tile(log_regs, len(log_lrs)), values, color='red', s=50, label='原始数据点')
    
    # 设置坐标轴标签
    ax.set_xlabel('log10(学习率)')
    ax.set_ylabel('log10(正则化参数)')
    ax.set_zlabel('第五轮验证准确率')
    
    # 设置标题
    ax.set_title('超参数对第五轮验证准确率的影响（插值表面图）')
    
    # 添加颜色条
    fig.colorbar(surf, ax=ax, label='验证准确率')
    
    # 添加图例
    ax.legend()
    
    # 保存图像
    if not os.path.exists(save_path):
        os.makedirs(f"{save_path}/Hyperparameter_search01.png")  # 如果目录不存在则创建
    plt.savefig(f"{save_path}/Hyperparameter_search01.png", dpi=300, bbox_inches='tight')
    logging.info(f"已保存三维插值图像到: {save_path}")
    
    # 关闭图形，防止显示
    plt.close(fig)