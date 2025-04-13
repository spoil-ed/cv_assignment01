import numpy as np
from tqdm import tqdm
from numba import jit
import logging

def relu(Z):
    return np.maximum(0, Z)

@jit(nopython=True)
def relu_derivative(Z):
    return Z > 0

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

@jit(nopython=True)
def conv_forward(X, W, b, stride=1, padding=0):
    N, H_in, W_in, C_in = X.shape
    F, kH, kW, _ = W.shape
    H_out = (H_in + 2 * padding - kH) // stride + 1
    W_out = (W_in + 2 * padding - kW) // stride + 1
    out = np.zeros((N, H_out, W_out, F))

    # 如果有 padding，手动填充（Numba 不支持 np.pad）
    if padding > 0:
        X_padded = np.zeros((N, H_in + 2 * padding, W_in + 2 * padding, C_in))
        X_padded[:, padding:-padding, padding:-padding, :] = X
    else:
        X_padded = X

    for n in range(N):
        for f in range(F):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride
                    h_end = h_start + kH
                    w_start = j * stride
                    w_end = w_start + kW
                    # 使用 NumPy 数组操作加速
                    window = X_padded[n, h_start:h_end, w_start:w_end, :]
                    out[n, i, j, f] = np.sum(window * W[f]) + b[f, 0, 0]

    return out

@jit(nopython=True)
def conv_backward(d_out, X, W, b, stride=1, padding=0):
    N, H_in, W_in, C = X.shape
    F, kH, kW, _ = W.shape
    H_out, W_out = d_out.shape[1], d_out.shape[2]
    dX = np.zeros_like(X)
    dW = np.zeros_like(W)
    db = np.zeros_like(b)

    if padding > 0:
        X_padded = np.zeros((N, H_in + 2 * padding, W_in + 2 * padding, C))
        X_padded[:, padding:-padding, padding:-padding, :] = X
        dX_padded = np.zeros_like(X_padded)
    else:
        X_padded = X
        dX_padded = dX

    # 计算偏置梯度
    for f in range(F):
        db[f, 0, 0] = np.sum(d_out[:, :, :, f])

    # 计算权重和输入梯度
    for n in range(N):
        for f in range(F):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride
                    h_end = h_start + kH
                    w_start = j * stride
                    w_end = w_start + kW
                    dW[f] += d_out[n, i, j, f] * X_padded[n, h_start:h_end, w_start:w_end, :]
                    dX_padded[n, h_start:h_end, w_start:w_end, :] += d_out[n, i, j, f] * W[f]

    if padding > 0:
        dX = dX_padded[:, padding:-padding, padding:-padding, :]
    else:
        dX = dX_padded

    return dX, dW, db

@jit(nopython=True)
def batch_norm_forward(x, gamma, beta, eps=1e-5):
    
    mu = np.sum(x, axis=0)/len(x)
    var = np.sum(x*x, axis=0)/len(x) - mu*mu
    x_norm = (x - mu) / np.sqrt(var + eps)
    out = gamma * x_norm + beta
    
    return out, x_norm, mu, var

@jit(nopython=True)
def batch_norm_backward(dout, gamma, x_norm, mu, var, x, eps=1e-5):
    N, H, W, C = dout.shape
    
    dgamma = np.sum(dout * x_norm, axis=0)
    dbeta = np.sum(dout, axis=0)
    
    dx_norm = dout * gamma
    dvar = np.sum(dx_norm * (x - mu) * -0.5 * (var + eps)**(-1.5), axis=0)
    dmu = np.sum(dx_norm * -1 / np.sqrt(var + eps), axis=0) + dvar * np.sum(-2 * (x - mu), axis=0) / len(x)
    dx = dx_norm / np.sqrt(var + eps) + dvar * 2 * (x - mu) / (N * H * W) + dmu / (N * H * W)
    
    return dx, dgamma, dbeta

@jit(nopython=True)
def pool_forward(X, pool_size=2, stride=2):
        # 最大池化前向传播
        N, Height, Width, C = X.shape
        H_out = (Height - pool_size) // stride + 1
        W_out = (Width - pool_size) // stride + 1
        out = np.zeros((N, H_out, W_out, C))
        pool_mask = np.zeros_like(X)
        
        for n in range(N):
            for c in range(C):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * stride
                        h_end = h_start + pool_size
                        w_start = w * stride
                        w_end = w_start + pool_size
                        
                        window = X[n, h_start:h_end, w_start:w_end, c]

                        out[n, h, w, c] = np.max(window)
                        max_idx = np.argmax(window) #扁平化索引
                        pool_mask[n, h_start + max_idx // pool_size, w_start + max_idx % pool_size, c] = 1
        return out, pool_mask

@jit(nopython=True)
def pool_backward(d_out, X, pool_mask, pool_size=2, stride=2):
        N, H_in, W_in, C = X.shape
        _, H_out, W_out, _ = d_out.shape
        
        # 初始化输入梯度
        dX = np.zeros_like(X)
        
        # 反向传播
        for n in range(N):
            for c in range(C):
                for h in range(H_out):
                    for w in range(W_out):
                        # 计算池化窗口位置
                        h_start = h * stride
                        h_end = h_start + pool_size
                        w_start = w * stride
                        w_end = w_start + pool_size
                        
                        # 将输出梯度分配到最大值位置
                        window_mask = pool_mask[n, h_start:h_end, w_start:w_end, c]
                        # print(f"dX:{dX.shape}, d_out: {d_out.shape}, window_mask: {window_mask.shape}")
                        # print(f"dX:{dX[n, h_start:h_end, w_start:w_end, c].shape}, d_out: {d_out[n, h, w, c].shape}, window_mask: {window_mask.shape}")
                        dX[n, h_start:h_end, w_start:w_end, c] += d_out[n, h, w, c] * window_mask
        
        return dX

@jit(nopython=True)
def fc_forward(X, W, b):
        Z = np.dot(X, W) + b
        return Z

@jit(nopython=True)
def fc_backward(dZ, X, W):
    dW = np.dot(X.T, dZ) / X.shape[0]
    db = np.sum(dZ, axis=0) / X.shape[0]
    db = db.reshape(1, -1)
    dX = np.dot(dZ, W.T) / X.shape[0]
    return dX, dW, db

class Conv3LayerNN:
    def __init__(self, input_shape=(32, 32, 3), conv1_filters=32, conv2_filters=32, fc_size=128, num_classes=10):
        # 输入形状: (H, W, C) = (32, 32, 3)
        self.input_shape = input_shape

        # 对于卷积层 (He 初始化)
        # 第一层卷积: conv1_filters 个 3x3 卷积核，步幅 1，无填充
        self.W1 = np.random.randn(conv1_filters, 3, 3, 3) * np.sqrt(2.0 / (3 * 3 * 3))  # [filters, kH, kW, in_channels]
        self.b1 = np.zeros((conv1_filters, 1, 1))
        self.conv1_out_shape = (30, 30, conv1_filters)  # 32-3+1=30
        self.pool1_out_shape = (15, 15, conv1_filters)  # 30/2=15
        self.gamma1 = np.ones(self.conv1_out_shape)  # BN 参数
        self.beta1 = np.zeros(self.conv1_out_shape)
        self.bn_cache1 = {}

        # 第二层卷积: conv2_filters 个 3x3 卷积核，步幅 1，无填充
        self.W2 = np.random.randn(conv2_filters, 3, 3, conv1_filters) * np.sqrt(2.0 / (3 * 3 * conv1_filters))
        self.b2 = np.zeros((conv2_filters, 1, 1))
        self.conv2_out_shape = (13, 13, conv2_filters)  # 15-3+1=13
        self.pool2_out_shape = (6, 6, conv2_filters)   # 13/2=6 (向下取整)
        self.gamma2 = np.ones(self.conv2_out_shape)  # BN 参数
        self.beta2 = np.zeros(self.conv2_out_shape)
        self.bn_cache2 = {}

        # 全连接层: 输入为池化后的展平特征，输出为 num_classes
        

        fc_input_size = 6 * 6 * conv2_filters

        self.W3 = np.random.randn(fc_input_size, num_classes) * np.sqrt(2.0 / fc_input_size)
        self.b3 = np.zeros((1, num_classes))

    def forward(self, X):

        # 前向传播
        self.X = X  # [N, 32, 32, 3]
        self.conv1 = conv_forward(X, self.W1, self.b1)  # [N, 30, 30, 16]
        self.bn1, self.x_norm1, self.mu1, self.var1 = batch_norm_forward(self.conv1, self.gamma1, self.beta1)
        self.relu1 = relu(self.bn1)
        self.pool1, self.pool_mask1 = pool_forward(self.relu1)           # [N, 15, 15, 16]

        self.conv2 = conv_forward(self.pool1, self.W2, self.b2)  # [N, 13, 13, 32]
        self.bn2, self.x_norm2, self.mu2, self.var2 = batch_norm_forward(self.conv2, self.gamma2, self.beta2)
        self.relu2 = relu(self.bn2)
        self.pool2, self.pool_mask2 = pool_forward(self.relu2)           # [N, 6, 6, 32]
        # 展平并全连接
        N = X.shape[0]
        self.pool2_flat = self.pool2.reshape(N, -1)  # [N, 6*6*32]
        self.scores = fc_forward(self.pool2_flat, self.W3, self.b3)  # [N, num_classes]

        # Softmax
        self.probs = softmax(self.scores)  # [N, num_classes]
        
        return self.probs

    def backward(self, X, y, probs, reg_lambda):
        N = X.shape[0]

        # Softmax 梯度
        delta3 = probs.copy()
        delta3[range(N), y] -= 1 # 区分正确预测与错误预测
        
        # 全连接层 1 梯度
        delta2_flat, dW3, db3 = fc_backward(delta3, self.pool2_flat, self.W3)
        delta2 = delta2_flat.reshape(self.pool2.shape)
        
        # 池化层 2 反向传播
        dpool2 = pool_backward(delta2, self.relu2, self.pool_mask2)
        drelu2 = dpool2 * relu_derivative(self.bn2)
        dbn2, dgamma2, dbeta2 = batch_norm_backward(drelu2, self.gamma2, self.x_norm2, self.mu2, self.var2, self.conv2)
        dconv2, dW2, db2 = conv_backward(dbn2, self.pool1, self.W2, self.b2)
        db2 = db2.reshape(self.b2.shape)    
        # logging.info(f"dpool2 max: {np.max(dpool2)}, drelu2 max: {np.max(drelu2)}, dbn2 max: {np.max(dbn2)}, dconv2 max: {np.max(dconv2)}, dW2 max: {np.max(dW2)}, db2 max: {np.max(db2)}")

        # 池化层 1 反向传播
        dpool1 = pool_backward(dconv2, self.relu1, self.pool_mask1)
        drelu1 = dpool1 * relu_derivative(self.bn1)
        dbn1, dgamma1, dbeta1 = batch_norm_backward(drelu1, self.gamma1, self.x_norm1, self.mu1, self.var1, self.conv1)
        dconv1, dW1, db1 = conv_backward(dbn1, X, self.W1, self.b1)
        # 卷积层 1 梯度
        db1 = db1.reshape(self.b1.shape)
        
        # logging.info(f"W1 norm: {np.linalg.norm(self.W1)}, W2 norm: {np.linalg.norm(self.W2)}, W3 norm: {np.linalg.norm(self.W3)}")
        # logging.info(f"backward: dW1 max={np.max(dW1)}, dW2 max={np.max(dW2)}, dW3 max={np.max(dW3)}")
        # tqdm.write(f"backward: dW1 max={np.max(dW1)}, dW2 max={np.max(dW2)}, dW3 max={np.max(dW3)}")
        return dW1, db1, dW2, db2, dW3, db3, dgamma1, dbeta1, dgamma2, dbeta2
        

    def compute_loss(self, probs, y, reg_lambda):
        N = probs.shape[0]

        epsilon = 1e-10
        probs = np.clip(probs, epsilon, 1 - epsilon)
        correct_logprobs = -np.log(probs[range(N), y])
        data_loss = np.sum(correct_logprobs) / N
        reg_loss = 0.5 * reg_lambda * (np.sum(self.W1**2) + np.sum(self.W2**2) + np.sum(self.W3**2))
        return data_loss + reg_loss

    def update_params(self, grads, lr=0.01, reg=0.001, max_grad_norm=5.0):
        dW1, db1, dW2, db2, dW3, db3, dgamma1, dbeta1, dgamma2, dbeta2 = grads

        dW1 += reg * self.W1
        dW2 += reg * self.W2
        dW3 += reg * self.W3

        for grad in [dW1, dW2, dW3, dgamma1, dbeta1, dgamma2, dbeta2]:
            norm = np.linalg.norm(grad)
            if norm > max_grad_norm:
                grad *= max_grad_norm / norm

        self.W1 -= lr * dW1
        # print(f"self.b1: {self.b1.shape},db1: {db1.shape}")
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W3 -= lr * dW3
        self.b3 -= lr * db3
        # print(f"self.gamma1: {self.gamma1},dgamma1: {dgamma1}")
        self.gamma1 -= lr * dgamma1
        self.beta1 -= lr * dbeta1
        self.gamma2 -= lr * dgamma2
        self.beta2 -= lr * dbeta2