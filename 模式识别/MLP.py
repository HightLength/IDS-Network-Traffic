import torch
import numpy as np
import matplotlib.pyplot as plt


plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
class Perceptron:
    def __init__(self, input_size, learning_rate=1.0):
        self.weights = None
        self.learning_rate = learning_rate
        self.input_size = input_size

    def train(self, X, y, max_epochs=1000):
        # 初始化权重，按照题目要求 w(1) = (1, -2, -2, 0)^T
        # 注意：这里使用增广权重向量 [w0, w1, w2, w3]，其中w0是偏置项
        self.weights = torch.tensor([0, 0, 0, 0.0], dtype=torch.float32)

        print("初始权重:", self.weights.numpy())

        errors = []
        for epoch in range(max_epochs):
            total_error = 0
            for i in range(len(X)):
                # 前向传播
                prediction = self.predict_single(X[i])

                # 计算误差
                error = y[i] - prediction

                if error != 0:
                    # 更新权重
                    self.weights += self.learning_rate * error * X[i]

                    total_error += 1
                print(f"向量:{X[i].numpy()},真实类别:{y[i]},预测类别:{prediction},权重更新:{self.weights.numpy()}")
            errors.append(total_error)

            print(f"Epoch {epoch + 1}: 错误数 = {total_error}, 权重 = {self.weights.numpy()}")

            # 如果没有错误，停止训练
            if total_error == 0:
                print(f"\n训练完成! 共 {epoch + 1} 个周期")
                break

        return errors

    def predict_single(self, x):
        # 计算净输入
        net_input = torch.dot(self.weights, x)
        # 使用符号函数作为激活函数
        return 1 if net_input > 0 else -1

    def predict(self, X):
        predictions = []
        for x in X:
            predictions.append(self.predict_single(x))
        return torch.tensor(predictions)


def main():
    # 定义模式类别
    # ω1类样本
    omega1_samples = [
        [0, 0, 0],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0]
    ]

    # ω2类样本
    omega2_samples = [
        [0, 0, 1],
        [0, 1, 1],
        [0, 1, 0],
        [1, 1, 1]
    ]

    # 准备训练数据（增广形式，添加偏置项1）
    X_omega1 = [torch.tensor([1.0] + sample, dtype=torch.float32) for sample in omega1_samples]
    X_omega2 = [torch.tensor([1.0] + sample, dtype=torch.float32) for sample in omega2_samples]

    # 合并所有样本
    X = X_omega1 + X_omega2

    # 准备标签：ω1类为1，ω2类为-1
    y = torch.tensor([1, 1, 1, 1, -1, -1, -1, -1], dtype=torch.float32)

    print("=== 感知器算法模式分类 ===")
    print("ω1类样本:")
    for i, sample in enumerate(omega1_samples):
        print(f"  样本{i + 1}: {sample}")

    print("\nω2类样本:")
    for i, sample in enumerate(omega2_samples):
        print(f"  样本{i + 1}: {sample}")

    print(f"\n初始权重: w = [-1, -2, -2, 0]")

    # 创建并训练感知器
    perceptron = Perceptron(input_size=4, learning_rate=1.0)
    errors = perceptron.train(X, y)

    print(f"\n=== 最终结果 ===")
    print(f"解向量: w = {perceptron.weights.numpy()}")
    print(
        f"决策函数: d(x) = {perceptron.weights[0]:.1f} + {perceptron.weights[1]:.1f}x1 + {perceptron.weights[2]:.1f}x2 + {perceptron.weights[3]:.1f}x3")

    # 测试分类效果
    predictions = perceptron.predict(X)
    print(f"\n=== 分类结果验证 ===")
    for i, (x, true_label, pred) in enumerate(zip(X, y, predictions)):
        sample_type = "ω1" if true_label == 1 else "ω2"
        status = "正确" if true_label == pred else "错误"
        print(
            f"样本{i + 1}({sample_type}): {x[1:].numpy()} -> 预测: {pred.item():.0f}, 实际: {true_label.item():.0f} [{status}]")

    # 可视化训练过程
    plt.figure(figsize=(12, 5))

    # 错误数变化
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(errors) + 1), errors, 'b-o', linewidth=2, markersize=6)
    plt.xlabel('训练周期')
    plt.ylabel('错误分类数')
    plt.title('感知器训练过程')
    plt.grid(True)

    # 3D可视化分类结果
    ax = plt.subplot(1, 2, 2, projection='3d')

    # 绘制ω1类样本
    omega1_array = np.array(omega1_samples)
    ax.scatter(omega1_array[:, 0], omega1_array[:, 1], omega1_array[:, 2],
               c='red', marker='o', s=100, label='ω1', alpha=0.7)

    # 绘制ω2类样本
    omega2_array = np.array(omega2_samples)
    ax.scatter(omega2_array[:, 0], omega2_array[:, 1], omega2_array[:, 2],
               c='blue', marker='^', s=100, label='ω2', alpha=0.7)

    # 绘制决策平面
    w = perceptron.weights.numpy()

    # 创建网格
    xx, yy = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))

    # 计算z值 (x3)
    if w[3] != 0:  # 确保分母不为零
        zz = (-w[0] - w[1] * xx - w[2] * yy) / w[3]
        ax.plot_surface(xx, yy, zz, alpha=0.3, color='green')
    else:
        print("警告: w[3] = 0，无法绘制决策平面")

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')
    ax.set_title('模式分类结果')
    ax.legend()

    plt.tight_layout()
    plt.show()

    # 打印决策边界方程
    print(f"\n=== 决策平面方程 ===")
    if w[3] != 0:
        print(f"{w[1]:.1f}x1 + {w[2]:.1f}x2 + {w[3]:.1f}x3 + {w[0]:.1f} = 0")
        print(f"或者:")
        print(f"x3 = {-w[0] / w[3]:.1f} + {-w[1] / w[3]:.1f}x1 + {-w[2] / w[3]:.1f}x2")
    else:
        print(f"{w[1]:.1f}x1 + {w[2]:.1f}x2 + {w[0]:.1f} = 0")


if __name__ == "__main__":
    main()