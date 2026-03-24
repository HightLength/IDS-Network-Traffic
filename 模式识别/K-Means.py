import torch
import matplotlib.pyplot as plt


plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
class KMeans:
    def __init__(self, k=2, max_iters=100, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None
        self.labels = None

    def fit(self, X):

        # 使用K-means算法对数据进行聚类
        # X: 输入数据, shape (n_samples, n_features)
        n_samples, n_features = X.shape

        # 随机初始化聚类中心
        random_indices = torch.randperm(n_samples)[:self.k]
        self.centroids = X[random_indices].clone()

        for iteration in range(self.max_iters):
            # 计算每个点到聚类中心的距离
            distances = self._compute_distances(X)

            # 分配标签到最近的聚类中心
            self.labels = torch.argmin(distances, dim=1)

            # 保存旧的聚类中心检查是否收敛
            old_centroids = self.centroids.clone()

            # 更新聚类中心
            for i in range(self.k):
                # 找到属于当前簇的所有点
                mask = self.labels == i
                if mask.sum() > 0:  # 确保簇不为空
                    self.centroids[i] = X[mask].mean(dim=0)

            # 检查收敛
            centroid_shift = torch.norm(self.centroids - old_centroids, dim=1).max()
            if centroid_shift < self.tol:
                print(f"在迭代 {iteration + 1} 后收敛")
                break

        return self.labels, self.centroids

    def _compute_distances(self, X):

    #计算每个点到所有聚类中心的欧氏距离

        distances = torch.zeros((X.shape[0], self.k))
        for i in range(self.k):
            # 计算欧氏距离
            diff = X - self.centroids[i]
            distances[:, i] = torch.sqrt(torch.sum(diff ** 2, dim=1))
        return distances

    def predict(self, X):
    #预测新数据的标签
        distances = self._compute_distances(X)
        return torch.argmin(distances, dim=1)


def plot_results(X, labels, centroids, title="K-means聚类结果"):
    #可视化聚类结果
    plt.figure(figsize=(10, 8))

    # 为每个簇使用不同的颜色
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown']

    for i in range(len(centroids)):
        # 绘制属于当前簇的点
        cluster_points = X[labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                    c=colors[i], label=f'簇 {i + 1}', alpha=0.7, s=100)

        # 绘制聚类中心
        plt.scatter(centroids[i, 0], centroids[i, 1],
                    c=colors[i], marker='*', s=300, edgecolors='black',
                    linewidth=2, label=f'中心 {i + 1}')

    plt.xlabel('X坐标')
    plt.ylabel('Y坐标')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def main():
    # 给定的数据点
    data_points = [
        [0, 0], [3, 8], [2, 2], [1, 1],
        [5, 3], [4, 8], [6, 3], [5, 4],
        [6, 4], [7, 5]
    ]

    # 转换为PyTorch张量
    X = torch.tensor(data_points, dtype=torch.float32)

    print("原始数据点:")
    for i, point in enumerate(data_points):
        print(f"x_{i + 1}{tuple(point)}")
    print()

    # 尝试不同的K值
    for k in [5]:
        print(f"=== K={k} 的聚类结果 ===")

        # 创建并训练K-means模型
        kmeans = KMeans(k=k)
        labels, centroids = kmeans.fit(X)

        # 输出结果
        print("聚类中心坐标:")
        for i, centroid in enumerate(centroids):
            print(f"中心 {i + 1}: ({centroid[0]:.2f}, {centroid[1]:.2f})")

        print("\n各点的簇分配:")
        for i, (point, label) in enumerate(zip(data_points, labels)):
            print(f"x_{i + 1}{tuple(point)} -> 簇 {label.item() + 1}")

        print("\n簇统计:")
        for i in range(k):
            cluster_size = (labels == i).sum().item()
            print(f"簇 {i + 1}: {cluster_size} 个点")

        print("-" * 50)

        # 可视化结果
        plot_results(X, labels, centroids, f"结果 (K={k})")


if __name__ == "__main__":
    main()