#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName    :origin1.py
# @Time        :2025/1/9 上午10:39
# @Author      :InubashiriLix

import numpy as np
import matplotlib.pyplot as plt

# 不考虑储层的大小、频谱半径、输入缩放以及储存库神经元激活函数
class EchoStateNetwork_1:
    def __init__(self, reservoir_size, spectral_radius=0.9):
        # 初始化网络参数
        self.reservoir_size = reservoir_size

        # 储层权重
        self.W_res = np.random.rand(reservoir_size, reservoir_size) - 0.5
        self.W_res *= spectral_radius / \
                      np.max(np.abs(np.linalg.eigvals(self.W_res)))

        # 输入权重
        self.W_in = np.random.rand(reservoir_size, 1) - 0.5

        # 输出权重（待训练）
        self.W_out = None

    def train(self, input_data, target_data):
        # 使用伪逆训练输出权重
        self.W_out = np.dot(np.linalg.pinv(self.run_reservoir(input_data)), target_data)

    def predict(self, input_data):
        # 使用训练好的输出权重进行预测
        return np.dot(self.run_reservoir(input_data), self.W_out)

    def run_reservoir(self, input_data):
        # 初始化储层状态
        reservoir_states = np.zeros((len(input_data), self.reservoir_size))

        # 运行储层
        for t in range(1, len(input_data)):
            reservoir_states[t, :] = np.tanh(
                np.dot(
                    self.W_res, reservoir_states[t - 1, :]) + np.dot(self.W_in, input_data[t])
            )

        return reservoir_states

# 生成合成数据（输入：随机噪音，目标：正弦波）
time = np.arange(0, 20, 0.1)
noise = 0.1 * np.random.rand(len(time))
sine_wave_target = np.sin(time)

# 创建Echo State网络
reservoir_size = 50
spectral_radius = 0.9
leaking_rate = 0.3
input_scaling = 1.0
activation_function = np.tanh

# 两种ESN，二选一
# 第一种ESN
esn = EchoStateNetwork_1(reservoir_size)

training_input = noise[:, None]
training_target = sine_wave_target[:, None]

# 训练ESN
esn.train(training_input, training_target)

# 生成测试数据（为简单起见与训练数据相似）
test_input = noise[:, None]

# 进行预测
predictions = esn.predict(test_input)

# 绘制结果
plt.figure(figsize=(10, 6))
plt.plot(time, sine_wave_target, label='True wave',
         linestyle='--', marker='o')
plt.plot(time, predictions, label='Predict wave', linestyle='--', marker='o')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()
plt.show()