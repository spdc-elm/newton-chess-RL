# -*- coding: utf-8 -*-
"""正方形棋盘的 D4（四旋转 × 镜像）训练增强。"""
import numpy as np

D4_COUNT = 8


def opening_orbit_representatives(size):
    """返回正方形棋盘在 D4 下的不等价首着 action（行主序代表）。"""
    if size < 1:
        raise ValueError("棋盘尺寸必须为正")
    def orbit(x, y):
        pts = set()
        for mirror in (False, True):
            xx = size - 1 - x if mirror else x
            yy = y
            for _ in range(4):
                pts.add(yy * size + xx)
                xx, yy = size - 1 - yy, xx
        return pts
    remaining = set(range(size * size))
    reps = []
    while remaining:
        a = min(remaining)
        pts = orbit(a % size, a // size)
        reps.append(a)
        remaining.difference_update(pts)
    return sorted(reps)


def _transform_spatial(x, transform):
    """变换数组最后两个空间轴；0..3 旋转，4..7 镜像后旋转。"""
    if not 0 <= transform < D4_COUNT:
        raise ValueError("D4 transform 必须在 0..7")
    if x.shape[-2] != x.shape[-1]:
        raise ValueError("D4 8 倍增强只支持正方形棋盘")
    y = np.flip(x, axis=-1) if transform >= 4 else x
    y = np.rot90(y, k=transform % 4, axes=(-2, -1))
    return np.ascontiguousarray(y)


def transform_obs_pi(obs, pi, transform):
    """同步变换单个/批量 obs 与行主序 policy。"""
    obs = np.asarray(obs)
    pi = np.asarray(pi)
    h, w = obs.shape[-2:]
    if h != w or pi.shape[-1] != h * w:
        raise ValueError("obs/pi 尺寸不匹配或棋盘不是正方形")
    pi_grid = pi.reshape(pi.shape[:-1] + (h, w))
    out_obs = _transform_spatial(obs, transform)
    out_pi = _transform_spatial(pi_grid, transform).reshape(pi.shape)
    return out_obs, np.ascontiguousarray(out_pi)


def expand_d4_batch(obs, pi, value):
    """每个基础样本展开为全部 8 个对称版本，value 保持不变。"""
    obs = np.asarray(obs)
    pi = np.asarray(pi)
    value = np.asarray(value)
    if obs.ndim != 4 or pi.ndim != 2 or value.ndim != 1:
        raise ValueError("期望 obs[B,C,H,W]、pi[B,H*W]、value[B]")
    if not (len(obs) == len(pi) == len(value)):
        raise ValueError("obs/pi/value batch 长度不一致")
    obs_parts, pi_parts = [], []
    for transform in range(D4_COUNT):
        o, p = transform_obs_pi(obs, pi, transform)
        obs_parts.append(o)
        pi_parts.append(p)
    return (
        np.ascontiguousarray(np.concatenate(obs_parts, axis=0)),
        np.ascontiguousarray(np.concatenate(pi_parts, axis=0)),
        np.ascontiguousarray(np.tile(value, D4_COUNT)),
    )
