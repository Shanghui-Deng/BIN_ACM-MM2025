import torch
from torch import nn


class BanzhafModule(nn.Module):
    def __init__(self, planes=512, num_heads=8):
        super().__init__()
        # 1D卷积替代原2D卷积
        self.cnn1 = nn.Conv1d(1, planes, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(planes)
        self.cnn2 = nn.Conv1d(planes, 1, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        # 使用标准多头注意力层
        self.attn = nn.MultiheadAttention(
            embed_dim=planes,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, x):
        """
        Args:
            x: [N, 1, N] 输入相似度矩阵
        Returns:
            [N, 1, N] 增强后的相似度矩阵
        """
        orig_shape = x.shape  # [N, 1, N]

        # 特征提取
        x = self.relu(self.bn(self.cnn1(x)))  # [N, planes, N]

        # 准备注意力输入
        attn_input = x.permute(0, 2, 1)  # [N, N, planes]

        # 注意力计算
        attn_output, _ = self.attn(
            query=attn_input,
            key=attn_input,
            value=attn_input,
            need_weights=False
        )  # [N, N, planes]

        # 残差连接
        x = x + attn_output.permute(0, 2, 1)  # [N, planes, N]

        # 最终输出
        x = self.cnn2(x)  # [N, 1, N]
        return x