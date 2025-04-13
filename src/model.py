import cupy as cp
from tqdm import tqdm
import logging
import numpy as np
from cupy.lib.stride_tricks import as_strided

# ReLU 和 Softmax 不需要 Numba，直接使用 CuPy
def relu(Z):
    return cp.maximum(0, Z)

def relu_derivative(Z):
    return Z > 0  # CuPy 支持布尔操作

def softmax(x):
    exp_x = cp.exp(x - cp.max(x, axis=1, keepdims=True))
    return exp_x / cp.sum(exp_x, axis=1, keepdims=True)

# 卷积前向传播（移除 Numba，CuPy 自带加速）
import cupy as cp
from cupy.lib.stride_tricks import as_strided

def conv_forward(X, W, b, stride=1, padding=0):
    """
    卷积前向传播，适配 CuPy 的向量化实现
    参数:
        X: 输入数据，形状 (N, H_in, W_in, C_in)
        W: 卷积核，形状 (F, kH, kW, C_in)
        b: 偏置，形状 (F,) 或 (F, 1, 1)
        stride: 步幅，默认为 1
        padding: 填充，默认为 0
    返回:
        out: 输出，形状 (N, H_out, W_out, F)
    """
    # 获取维度
    N, H_in, W_in, C_in = X.shape
    F, kH, kW, C_in_filters = W.shape
    assert C_in == C_in_filters, "输入通道数和卷积核通道数不匹配"

    # 计算输出尺寸
    H_out = (H_in + 2 * padding - kH) // stride + 1
    W_out = (W_in + 2 * padding - kW) // stride + 1
    assert H_out > 0 and W_out > 0, "卷积参数导致输出尺寸无效"

    # 添加填充
    if padding > 0:
        X_padded = cp.pad(X, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode='constant')
    else:
        X_padded = X

    # 提取所有卷积窗口
    shape = (N, H_out, W_out, kH, kW, C_in)
    strides = (X_padded.strides[0], 
               X_padded.strides[1] * stride, 
               X_padded.strides[2] * stride, 
               X_padded.strides[1], 
               X_padded.strides[2], 
               X_padded.strides[3])
    windows = as_strided(X_padded, shape=shape, strides=strides)

    # 执行卷积（向量化）
    out = cp.tensordot(windows, W, axes=([3, 4, 5], [1, 2, 3]))  # 对 kH, kW, C_in 求和
    
    # 调整偏置形状并添加
    if b.ndim == 1:
        b = b.reshape(1, 1, 1, F)  # 从 (F,) 调整为 (1, 1, 1, F)
    elif b.shape != (1, 1, 1, F):
        b = b.reshape(1, 1, 1, F)  # 确保形状正确
    out += b  # 现在可以正确广播

    return out

# 卷积反向传播（移除 Numba）
import cupy as cp
from cupy.lib.stride_tricks import as_strided

def conv_backward(d_out, X, W, b, stride=1, padding=0):
    # 获取维度
    N, H_in, W_in, C = X.shape
    F, kH, kW, C_filters = W.shape
    assert C == C_filters, "输入通道数和卷积核通道数不匹配"
    H_out, W_out = d_out.shape[1], d_out.shape[2]

    # 初始化梯度
    dX = cp.zeros_like(X)
    dW = cp.zeros_like(W)
    db = cp.zeros_like(b)

    # 添加填充
    if padding > 0:
        X_padded = cp.pad(X, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode='constant')
        dX_padded = cp.zeros_like(X_padded)
    else:
        X_padded = X
        dX_padded = dX

    # 计算 db（偏置梯度）
    db = cp.sum(d_out, axis=(0, 1, 2)).reshape(1, 1, 1, F)  # 沿着 N, H_out, W_out 求和

    # 提取滑动窗口
    shape = (N, H_out, W_out, kH, kW, C)
    strides = (X_padded.strides[0], 
               X_padded.strides[1] * stride, 
               X_padded.strides[2] * stride, 
               X_padded.strides[1], 
               X_padded.strides[2], 
               X_padded.strides[3])
    windows = as_strided(X_padded, shape=shape, strides=strides)

    # 计算 dW（卷积核梯度）
    dW = cp.tensordot(d_out, windows, axes=([0, 1, 2], [0, 1, 2]))  # 沿着 N, H_out, W_out 求和

    # 计算 dX_padded（输入梯度）
    # 需要将 d_out 与 W 卷积，考虑 stride 和 padding
    W_rotated = cp.flip(W, axis=(1, 2))  # 旋转卷积核 180 度
    d_out_padded = cp.pad(d_out, ((0, 0), (kH-1, kH-1), (kW-1, kW-1), (0, 0)), mode='constant')
    shape_dx = (N, H_in + 2*padding - kH + 1, W_in + 2*padding - kW + 1, kH, kW, F)
    strides_dx = (d_out_padded.strides[0], 
                  d_out_padded.strides[1] * stride, 
                  d_out_padded.strides[2] * stride, 
                  d_out_padded.strides[1], 
                  d_out_padded.strides[2], 
                  d_out_padded.strides[3])
    windows_dx = as_strided(d_out_padded, shape=shape_dx, strides=strides_dx)
    dX_padded = cp.tensordot(windows_dx, W_rotated, axes=([3, 4, 5], [1, 2, 0]))  # 注意 W 的轴调整

    # 裁剪 dX（移除填充）
    if padding > 0:
        dX = dX_padded[:, padding:-padding, padding:-padding, :]
    else:
        dX = dX_padded

    return dX, dW, db

# 批归一化（移除 Numba）
def batch_norm_forward(x, gamma, beta, eps=1e-5):
    mu = cp.sum(x, axis=0) / len(x)
    var = cp.sum(x * x, axis=0) / len(x) - mu * mu
    x_norm = (x - mu) / cp.sqrt(var + eps)
    out = gamma * x_norm + beta
    return out, x_norm, mu, var

def batch_norm_backward(dout, gamma, x_norm, mu, var, x, eps=1e-5):
    N, H, W, C = dout.shape
    
    dgamma = cp.sum(dout * x_norm, axis=0)
    dbeta = cp.sum(dout, axis=0)
    
    dx_norm = dout * gamma
    dvar = cp.sum(dx_norm * (x - mu) * -0.5 * (var + eps)**(-1.5), axis=0)
    dmu = cp.sum(dx_norm * -1 / cp.sqrt(var + eps), axis=0) + dvar * cp.sum(-2 * (x - mu), axis=0) / len(x)
    dx = dx_norm / cp.sqrt(var + eps) + dvar * 2 * (x - mu) / (N * H * W) + dmu / (N * H * W)
    
    return dx, dgamma, dbeta

# 池化层（移除 Numba）
def pool_forward(X, pool_size=2, stride=2):
    N, Height, Width, C = X.shape
    H_out = (Height - pool_size) // stride + 1
    W_out = (Width - pool_size) // stride + 1

    # 初始化输出和掩码
    out = cp.zeros((N, H_out, W_out, C))
    pool_mask = cp.zeros_like(X)

    # 提取所有池化窗口
    shape = (N, H_out, W_out, pool_size, pool_size, C)
    strides = (X.strides[0], 
               X.strides[1] * stride, 
               X.strides[2] * stride, 
               X.strides[1], 
               X.strides[2], 
               X.strides[3])
    windows = cp.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)

    # 计算最大值和掩码
    out = cp.max(windows, axis=(3, 4))  # 沿着池化窗口维度取最大值
    max_indices = cp.argmax(windows, axis=(3, 4))  # 获取最大值索引

    # 将最大值索引转换为 pool_mask
    h_idx = max_indices // pool_size  # 行偏移
    w_idx = max_indices % pool_size   # 列偏移
    n_idx, h_out_idx, w_out_idx, c_idx = cp.ogrid[:N, :H_out, :W_out, :C]
    h_pos = h_out_idx * stride + h_idx
    w_pos = w_out_idx * stride + w_idx
    pool_mask[n_idx, h_pos, w_pos, c_idx] = 1

    return out, pool_mask

def pool_backward(d_out, X, pool_mask, pool_size=2, stride=2):
    N, H_in, W_in, C = X.shape
    _, H_out, W_out, _ = d_out.shape

    # 初始化输入梯度
    dX = cp.zeros_like(X)

    # 使用掩码直接分配梯度
    for h in range(H_out):
        for w in range(W_out):
            h_start = h * stride
            h_end = h_start + pool_size
            w_start = w * stride
            w_end = w_start + pool_size
            dX[:, h_start:h_end, w_start:w_end, :] += (
                d_out[:, h:h+1, w:w+1, :] * pool_mask[:, h_start:h_end, w_start:w_end, :]
            )

    return dX

# 全连接层（移除 Numba）
def fc_forward(X, W, b):
    Z = cp.dot(X, W) + b
    return Z

def fc_backward(dZ, X, W):
    dW = cp.dot(X.T, dZ) / X.shape[0]
    db = cp.sum(dZ, axis=0) / X.shape[0]
    db = db.reshape(1, -1)
    dX = cp.dot(dZ, W.T) / X.shape[0]
    return dX, dW, db

# Conv3LayerNN 类保持不变，但初始化和计算使用 CuPy
class Conv3LayerNN:
    def __init__(
        self,
        input_shape=(32, 32, 3),
        num_classes=10,
        conv1_filters=32,
        kernel1_size=(3, 3),
        conv1_stride=1,
        conv1_padding=0,
        conv2_filters=32,
        kernel2_size=(3, 3),
        conv2_stride=1,
        conv2_padding=0,
        pool_size=2,
        pool_stride=2,
        bn_epsilon=1e-5,
    ):
        self.input_shape = input_shape
        H_in, W_in, C_in = input_shape
        self.kernel1_height, self.kernel1_width = kernel1_size
        self.kernel2_height, self.kernel2_width = kernel2_size

        self.conv1_stride = conv1_stride
        self.conv2_stride = conv2_stride
        self.conv1_padding = conv1_padding
        self.conv2_padding = conv2_padding
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        self.bn_epsilon = bn_epsilon

        # 卷积模块 1
        self.W1 = (
            cp.random.randn(conv1_filters, self.kernel1_height, self.kernel1_width, C_in)
            * cp.sqrt(2.0 / (self.kernel1_height * self.kernel1_width * C_in))
        )
        self.b1 = cp.zeros((conv1_filters, 1, 1))
        conv1_H_out = (H_in + 2 * conv1_padding - self.kernel1_height) // conv1_stride + 1
        conv1_W_out = (W_in + 2 * conv1_padding - self.kernel1_width) // conv1_stride + 1
        self.conv1_out_shape = (conv1_H_out, conv1_W_out, conv1_filters)

        self.pool1_out_shape = (
            conv1_H_out // pool_stride,
            conv1_W_out // pool_stride,
            conv1_filters,
        )
        self.gamma1 = cp.ones(self.conv1_out_shape)
        self.beta1 = cp.zeros(self.conv1_out_shape)

        # 卷积模块 2
        self.W2 = (
            cp.random.randn(
                conv2_filters, self.kernel2_height, self.kernel2_width, conv1_filters
            )
            * cp.sqrt(2.0 / (self.kernel2_height * self.kernel2_width * conv1_filters))
        )
        self.b2 = cp.zeros((conv2_filters, 1, 1))
        pool1_H, pool1_W, _ = self.pool1_out_shape
        conv2_H_out = (pool1_H + 2 * conv2_padding - self.kernel2_height) // conv2_stride + 1
        conv2_W_out = (pool1_W + 2 * conv2_padding - self.kernel2_width) // conv2_stride + 1
        self.conv2_out_shape = (conv2_H_out, conv2_W_out, conv2_filters)
        self.pool2_out_shape = (
            conv2_H_out // pool_stride,
            conv2_W_out // pool_stride,
            conv2_filters,
        )
        self.gamma2 = cp.ones(self.conv2_out_shape)
        self.beta2 = cp.zeros(self.conv2_out_shape)

        fc_input_size = self.pool2_out_shape[0] * self.pool2_out_shape[1] * conv2_filters
        self.W3 = cp.random.randn(fc_input_size, num_classes) * cp.sqrt(2.0 / fc_input_size)
        self.b3 = cp.zeros((1, num_classes))

    def forward(self, X):
        self.X = X
        self.conv1 = conv_forward(
            X, self.W1, self.b1, stride=self.conv1_stride, padding=self.conv1_padding
        )
        self.bn1, self.x_norm1, self.mu1, self.var1 = batch_norm_forward(
            self.conv1, self.gamma1, self.beta1, eps=self.bn_epsilon
        )
        self.relu1 = relu(self.bn1)
        self.pool1, self.pool_mask1 = pool_forward(
            self.relu1, pool_size=self.pool_size, stride=self.pool_stride
        )
        self.conv2 = conv_forward(
            self.pool1, self.W2, self.b2, stride=self.conv2_stride, padding=self.conv2_padding
        )
        self.bn2, self.x_norm2, self.mu2, self.var2 = batch_norm_forward(
            self.conv2, self.gamma2, self.beta2, eps=self.bn_epsilon
        )
        self.relu2 = relu(self.bn2)
        self.pool2, self.pool_mask2 = pool_forward(
            self.relu2, pool_size=self.pool_size, stride=self.pool_stride
        )
        N = X.shape[0]
        self.pool2_flat = self.pool2.reshape(N, -1)
        self.scores = fc_forward(self.pool2_flat, self.W3, self.b3)
        self.probs = softmax(self.scores)
        return self.probs

    def backward(self, X, y, probs, reg_lambda):
        N = X.shape[0]
        delta3 = probs.copy()
        delta3[range(N), y] -= 1

        delta2_flat, dW3, db3 = fc_backward(delta3, self.pool2_flat, self.W3)
        delta2 = delta2_flat.reshape(self.pool2.shape)

        dpool2 = pool_backward(
            delta2, self.relu2, self.pool_mask2, pool_size=self.pool_size, stride=self.pool_stride
        )
        drelu2 = dpool2 * relu_derivative(self.bn2)
        dbn2, dgamma2, dbeta2 = batch_norm_backward(
            drelu2, self.gamma2, self.x_norm2, self.mu2, self.var2, self.conv2, eps=self.bn_epsilon
        )
        dconv2, dW2, db2 = conv_backward(
            dbn2, self.pool1, self.W2, self.b2, stride=self.conv2_stride, padding=self.conv2_padding
        )
        db2 = db2.reshape(self.b2.shape)

        dpool1 = pool_backward(
            dconv2, self.relu1, self.pool_mask1, pool_size=self.pool_size, stride=self.pool_stride
        )
        drelu1 = dpool1 * relu_derivative(self.bn1)
        dbn1, dgamma1, dbeta1 = batch_norm_backward(
            drelu1, self.gamma1, self.x_norm1, self.mu1, self.var1, self.conv1, eps=self.bn_epsilon
        )
        dconv1, dW1, db1 = conv_backward(
            dbn1, X, self.W1, self.b1, stride=self.conv1_stride, padding=self.conv1_padding
        )
        db1 = db1.reshape(self.b1.shape)

        return dW1, db1, dW2, db2, dW3, db3, dgamma1, dbeta1, dgamma2, dbeta2

    def compute_loss(self, probs, y, reg_lambda):
        N = probs.shape[0]
        epsilon = 1e-10
        probs = cp.clip(probs, epsilon, 1 - epsilon)
        correct_logprobs = -cp.log(probs[range(N), y])
        data_loss = cp.sum(correct_logprobs) / N
        reg_loss = 0.5 * reg_lambda * (cp.sum(self.W1**2) + cp.sum(self.W2**2) + cp.sum(self.W3**2))
        return data_loss + reg_loss

    def update_params(self, grads, lr=0.01, reg=0.001, max_grad_norm=5.0):
        dW1, db1, dW2, db2, dW3, db3, dgamma1, dbeta1, dgamma2, dbeta2 = grads

        dW1 += reg * self.W1
        dW2 += reg * self.W2
        dW3 += reg * self.W3

        for grad in [dW1, dW2, dW3, dgamma1, dbeta1, dgamma2, dbeta2]:
            norm = cp.linalg.norm(grad)
            if norm > max_grad_norm:
                grad *= max_grad_norm / norm

        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W3 -= lr * dW3
        self.b3 -= lr * db3
        self.gamma1 -= lr * dgamma1
        self.beta1 -= lr * dbeta1
        self.gamma2 -= lr * dgamma2
        self.beta2 -= lr * dbeta2

    def accuracy(self, probs, labels):
        acc = np.mean(np.argmax(probs, axis=1) == labels)
        return acc