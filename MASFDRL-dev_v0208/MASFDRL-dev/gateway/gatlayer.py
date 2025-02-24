import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GATLayer(nn.Module):
    def __init__(self, in_channels, out_channels, heads=1, dropout=0.5):
        """
        单层 GAT 实现。
        Args:
            in_channels: 输入特征的维度
            out_channels: 输出特征的维度
            heads: 注意力头的数量
            dropout: 丢弃率
        """
        super(GATLayer, self).__init__()
        self.gat = GATConv(in_channels, out_channels, heads=heads, dropout=dropout)

    def forward(self, x, edge_index):
        """
        前向传播。
        Args:
            x: 节点特征矩阵 [N, F]
            edge_index: 边索引矩阵 [2, M]
        Returns:
            更新后的节点特征 [N, F']
        """
        return self.gat(x, edge_index)


class HierarchicalGATLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1, dropout=0.5):
        """
        分层 GAT 实现，处理局部和全局层级的关系。
        Args:
            in_channels: 输入特征维度
            hidden_channels: 隐藏特征维度
            out_channels: 输出特征维度
            heads: 注意力头数量
            dropout: 丢弃率
        """
        super(HierarchicalGATLayer, self).__init__()
        self.local_gat = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.global_gat = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.combine = nn.Linear(hidden_channels * heads * 2, out_channels)

    def forward(self, x, edge_index_local, edge_index_global):
        """
        前向传播。
        Args:
            x: 节点特征矩阵 [N, F]
            edge_index_local: 局部邻接边索引 [2, M_local]
            edge_index_global: 全局邻接边索引 [2, M_global]
        Returns:
            更新后的节点特征 [N, F_out]
        """
        local_out = self.local_gat(x, edge_index_local)
        global_out = self.global_gat(x, edge_index_global)
        combined = torch.cat([local_out, global_out], dim=1)
        output = F.elu(self.combine(combined))
        return output
