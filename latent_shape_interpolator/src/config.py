import os
import torch
import random
import numpy as np


class DataConfiguration:
    GRID_SIZE = 36
    MIN_BOUND = -1.0
    MAX_BOUND = 1.0

    N_SURFACE_SAMPLING_RATIO = 0.3
    N_SURFACE_NOISY_SAMPLING_RATIO = 0.5
    N_VOLUME_SAMPLING_RATIO = 0.2

    N_TOTAL_SAMPLING = GRID_SIZE**3
    N_SURFACE_SAMPLING = int(N_TOTAL_SAMPLING * N_SURFACE_SAMPLING_RATIO)
    N_SURFACE_NOISY_SAMPLING = int(N_TOTAL_SAMPLING * N_SURFACE_NOISY_SAMPLING_RATIO)
    N_VOLUME_SAMPLING = N_TOTAL_SAMPLING - N_SURFACE_SAMPLING - N_SURFACE_NOISY_SAMPLING

    assert N_SURFACE_SAMPLING + N_SURFACE_NOISY_SAMPLING + N_VOLUME_SAMPLING == N_TOTAL_SAMPLING

    DATA_DIR = os.path.abspath(os.path.join(__file__, "../../data"))
    DATA_PATH = os.path.join(DATA_DIR, "03001627")
    DATA_PATH_PROCESSED = os.path.join(DATA_DIR, "processed")
    DATA_NAME = "models"
    DATA_NAME_OBJ = "model_normalized.obj"

    SCALE_MATRIX = [
        [0.95, 0.00, 0.00],
        [0.00, 0.95, 0.00],
        [0.00, 0.00, 1.00],
    ]

    WATERTIGHT_RESOLUTION = 20000

    NUM_LATENT_POINTS = 26


class ModelConfiguration:
    EPOCHS = 500
    SEED = 777

    BATCH_SIZE = 512
    ACCUMULATION_STEPS = 1

    ATTENTION_DIM = 256
    NUM_HEADS = 8
    HIDDEN_DIM = 512

    LR_LATENT_SHAPES = 1e-5
    LR_DECODER = 1e-3

    LATENT_POINTS_NOISE = 0.05

    LOSS_TRAIN_WEIGHT = 0.2
    LOSS_VALIDATION_WEIGHT = 1.0

    CLAMP = 0.1

    SAVE_NAME = "states.pth"

    SCHEDULER_PATIENCE = 5
    SCHEDULER_FACTOR = 0.1

    TRAIN_VALIDATION_RATIO = [0.8, 0.2]

    RECONSTRUCTION_INTERVAL = 1
    RECONSTRUCTION_WATERTIGHT_RESOLUTION = 50000
    RECONSTRUCTION_COUNT = 5
    RECONSTRUCTION_GRID_SIZE = 192

    MARCHING_CUBES_LEVEL = 0.00
    NUM_LAYERS = 5
    NUM_BLOCKS = 10
    
    K = 5

    OPTIMIZER = "AdamW"
    ACTIVATION = "ReLU"
    ACTIVATION_KWARGS = {"inplace": True}
    
    CUDA = "cuda"
    CPU = "cpu"

    DEVICE = CUDA
    if not torch.cuda.is_available():
        DEVICE = CPU

    print("CUDA status")
    print(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"  DEVICE: {DEVICE} \n")
    
    USE_MULTI_GPUS = False

    if DEVICE == CUDA:
        for i in range(torch.cuda.device_count()):
            print(f"Current device: {torch.cuda.get_device_name(i)}")
            
        USE_MULTI_GPUS = torch.cuda.device_count() >= 2


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

        print("\nSeeds status:")
        print(f"  Seeds set for torch        : {torch.initial_seed()}")
        print(f"  Seeds set for torch on GPU : {torch.cuda.initial_seed()}")
        print(f"  Seeds set for numpy        : {seed}")
        print(f"  Seeds set for random       : {seed} \n")

        Configuration.SEED = seed
