# -*- coding: utf-8 -*-
import random

import nf_env as env


class RandomBot:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def select_move(self, s):
        moves = env.legal_moves(s)
        return moves[self.rng.randrange(len(moves))]
