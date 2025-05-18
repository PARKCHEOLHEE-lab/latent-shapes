import os
import torch
import random
import numpy as np


class DataConfiguration:
    GRID_SIZE = 32
    MIN_BOUND = -1.0
    MAX_BOUND = 1.0

    DATA_DIR = os.path.abspath(os.path.join(__file__, "../../data"))
    DATA_PATH = os.path.join(DATA_DIR, "03001627")
    DATA_PATH_PROCESSED = os.path.join(DATA_DIR, "processed")
    DATA_NAME = "models"
    DATA_NAME_OBJ = "model_normalized.obj"

    SCALE_MATRIX = np.array(
        [
            [0.95, 0.00, 0.00],
            [0.00, 0.95, 0.00],
            [0.00, 0.00, 1.00],
        ]
    )

    WATERTIGHT_RESOLUTION = 70000

    NUM_LATENT_POINTS = 26
    GRID_SIZE_RECONSTRUCTION = 192


class ModelConfiguration:
    EPOCHS = 5000
    SEED = 777

    BATCH_SIZE = 64
    ACCUMULATION_STEPS = 16

    HIDDEN_DIM = 512

    LR_LATENT_POINTS = 1e-4
    LR_DECODER = 1e-5

    LOSS_TRAIN_WEIGHT = 0.5
    LOSS_VALIDATION_WEIGHT = 1.0

    CLAMP_VALUE = 0.1

    SAVE_NAME = "states.pth"

    SCHEDULER_PATIENCE = 5
    SCHEDULER_FACTOR = 0.1

    TRAIN_VALIDATION_RATIO = [0.8, 0.2]

    OPTIMIZER = "AdamW"

    DEVICE = "cuda"
    if not torch.cuda.is_available():
        DEVICE = "cpu"

    print("CUDA status")
    print(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"  DEVICE: {DEVICE} \n")


class Configuration(DataConfiguration, ModelConfiguration):
    def __init__(self):
        pass

    def to_dict(self):
        raw_config = {**vars(Configuration), **vars(ModelConfiguration), **vars(DataConfiguration)}
        config = {}
        for key, value in raw_config.items():
            if not key.startswith("__") and not callable(value):
                config[key] = value

        return config

    LOG_DIR_BASE = os.path.abspath(os.path.join(__file__, "../../runs"))

    @staticmethod
    def set_seed(seed: int = ModelConfiguration.SEED):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)

        print("Seeds status:")
        print(f"  Seeds set for torch        : {torch.initial_seed()}")
        print(f"  Seeds set for torch on GPU : {torch.cuda.initial_seed()}")
        print(f"  Seeds set for numpy        : {seed}")
        print(f"  Seeds set for random       : {seed} \n")

        Configuration.SEED = seed
