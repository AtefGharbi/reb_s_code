"""Two-sided CUSUM detector on SoC-increment residuals -- the 'reactive'
baseline referenced in Section 6.4 / [9] of the paper."""
import numpy as np


class CusumDetector:
    def __init__(self, k=0.003, h=0.012):
        self.k = k    # slack
        self.h = h    # alarm threshold
        self.g_pos = {}
        self.g_neg = {}

    def update(self, asset_id, residual):
        gp = max(0.0, self.g_pos.get(asset_id, 0.0) + residual - self.k)
        gn = max(0.0, self.g_neg.get(asset_id, 0.0) - residual - self.k)
        self.g_pos[asset_id] = gp
        self.g_neg[asset_id] = gn
        alarm = (gp > self.h) or (gn > self.h)
        if alarm:
            self.g_pos[asset_id] = 0.0
            self.g_neg[asset_id] = 0.0
        return alarm, max(gp, gn) / self.h  # alarm flag, normalized risk-like score
