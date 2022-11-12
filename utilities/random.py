import random

import torch


def initialize_random_numbers():
    seed_value = 922342341684579
    seed_generator = torch.Generator().manual_seed(seed_value)
    random.seed(seed_value)