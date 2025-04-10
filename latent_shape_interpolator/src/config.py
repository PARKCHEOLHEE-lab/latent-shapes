import os
import torch
import random
import numpy as np


class DataConfiguration:
    GRID_SIZE = 64
    MIN_BOUND = -1
    MAX_BOUND = 1

    DATA_DIR = ""
    DATA_PATH = ""


class ModelConfiguration:
    EPOCHS = 5000
    SEED = 777

    BATCH_SIZE = 64
    ACCUMULATION_STEPS = 8


class Configuration(ModelConfiguration):
    def __init__(self):
        pass

    def to_dict(self):
        raw_config = {**vars(Configuration), **vars(ModelConfiguration), **vars(DataConfiguration)}
        config = {}
        for key, value in raw_config.items():
            if not key.startswith("__") and not callable(value):
                config[key] = value

        return config

    LOG_DIR = os.path.abspath(os.path.join(__file__, "../../runs"))

    @staticmethod
    def set_seed(seed: int = ModelConfiguration.SEED):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)

        print("CUDA status")
        print(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")
        print(f"  DEVICE: {Configuration.DEVICE} \n")

        print("Seeds status:")
        print(f"  Seeds set for torch        : {torch.initial_seed()}")
        print(f"  Seeds set for torch on GPU : {torch.cuda.initial_seed()}")
        print(f"  Seeds set for numpy        : {seed}")
        print(f"  Seeds set for random       : {seed} \n")

        Configuration.SEED = seed
