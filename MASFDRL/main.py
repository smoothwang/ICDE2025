import torch
import matplotlib.pyplot as plt
from synthetic_data import generate_synthetic_data
from multi_agent_env import MultiAgentEnvironment
from train_loop import train_multi_agent
from policy_network import MultiAgentPolicy


def visualize_rewards(rewards_history, title="Training Performance"):
    """
    绘制训练奖励曲线。
    Args:
        rewards_history: 每回合的奖励历史记录
        title: 图表标题
    """
    plt.figure(figsize=(8, 6))
    plt.plot(rewards_history, label="Total Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()


def visualize_graph(graph_data):
    """
    可视化图结构及节点标签。
    Args:
        graph_data: PyTorch Geometric 格式的图数据
    """
    import networkx as nx
    G = nx.Graph()
    edge_index = graph_data.edge_index.numpy()
    for i in range(edge_index.shape[1]):
        G.add_edge(edge_index[0, i], edge_index[1, i])

    labels = graph_data.y.numpy()
    node_colors = ["red" if label == 1 else "green" for label in labels]

    plt.figure(figsize=(8, 8))
    nx.draw(G, node_color=node_colors, with_labels=True, node_size=100, font_size=8)
    plt.title("Graph Visualization")
    plt.show()


def main():
    # 参数配置
    num_nodes = 100
    num_edges = 300
    feature_dim = 5
    anomaly_ratio = 0.1
    num_agents = 4
    max_steps = 50
    num_episodes = 500
    gamma = 0.99
    lr = 0.001

    # 生成合成数据
    print("生成合成数据...")
    graph_data = generate_synthetic_data(num_nodes, num_edges, feature_dim, anomaly_ratio)

    # 初始化环境
    print("初始化多智能体环境...")
    env = MultiAgentEnvironment(
        num_nodes=graph_data.num_nodes,
        num_edges=graph_data.num_edges,
        feature_dim=graph_data.x.shape[1],
        num_agents=num_agents,
        max_steps=max_steps
    )
    env.graph = graph_data

    # 启动训练
    print("开始训练多智能体系统...")
    rewards_history = train_multi_agent(env, num_episodes=num_episodes, gamma=gamma, lr=lr)

    # 绘制奖励曲线
    visualize_rewards(rewards_history, title="Training on Synthetic Data")

    # 可视化图结构
    print("可视化图结构...")
    visualize_graph(graph_data)


if __name__ == "__main__":
    main()
