# -*- coding: utf-8 -*-
"""牛顿棋策略-价值网络（全卷积，尺寸无关）。

契约（docs/rl-interface.md）：
  输入  float32[(n+1), H, W]   —— nfEncodeCanonical 的输出
  输出  policy logits [B, H*W]（行主序，非法动作由外部掩码）
        value        [B]      —— 当前行动方视角的期望归一化外围分差，tanh 到 [-1,1]
全卷积 + 自适应池化 ⇒ 同一权重可在任意棋盘尺寸上推理。
"""
import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.act(self.c1(x))
        h = self.c2(h)
        return self.act(x + h)


class NFNet(nn.Module):
    def __init__(self, planes_in=3, channels=64, blocks=4):
        super().__init__()
        self.planes_in = planes_in
        self.channels = channels
        self.blocks_n = blocks
        self.stem = nn.Sequential(nn.Conv2d(planes_in, channels, 3, padding=1), nn.ReLU(inplace=True))
        self.trunk = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        # policy 头：1x1 卷积到每格一个 logit（尺寸无关）
        self.policy = nn.Sequential(
            nn.Conv2d(channels, 32, 1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )
        # value 头：全局池化后固定维度，同样尺寸无关
        self.value = nn.Sequential(
            nn.Conv2d(channels, 32, 1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(32, 1), nn.Tanh(),
        )
        self.apply(self._init)

    def reset_policy_head(self):
        """保留 trunk/value，仅重置 policy 头用于 Gumbel 路径依赖消融。"""
        self.policy.apply(self._init)

    def reset_value_head(self, zero_output=True):
        """切换 value 语义时重置 value 头；可令初始输出严格为 0。"""
        self.value.apply(self._init)
        if zero_output:
            nn.init.zeros_(self.value[4].weight)
            nn.init.zeros_(self.value[4].bias)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.stem(x)
        h = self.trunk(h)
        pol = self.policy(h).flatten(1)          # [B, H*W]
        val = self.value(h).squeeze(1)           # [B]
        return pol, val

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
