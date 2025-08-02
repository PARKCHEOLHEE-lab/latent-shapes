import os
import torch
import random
import trimesh
import numpy as np


class DataConfiguration:

    # bounds computed by `latent_shapes/src/bounds.py`
    MIN_X_BOUND = -0.6149350000
    MIN_Y_BOUND = -0.7674950000
    MIN_Z_BOUND = -0.6662140000
    MAX_X_BOUND = 0.6443440000
    MAX_Y_BOUND = 0.7928820000
    MAX_Z_BOUND = 0.5812350000
    
    N_SURFACE_SAMPLING_RATIO = 0.3
    N_VOLUME_SAMPLING_RATIO = 0.2
    N_SURFACE_NOISY_SAMPLING_RATIO = 0.5

    SURFACE_NOISY_SAMPLING_RANGE = 0.10

    GRID_SIZE = 36
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

    BOX_MESH_SCALE_MATRIX = [
        [1.50, 0.00, 0.00],
        [0.00, 1.50, 0.00],
        [0.00, 0.00, 1.10],
    ]

    WATERTIGHT_RESOLUTION = 20000

    LATENT_SHAPE_SUBDIVISION_COUNT = 2
    _box = trimesh.creation.box()
    for _ in range(LATENT_SHAPE_SUBDIVISION_COUNT):
        _box = _box.subdivide()

    NUM_LATENT_SHAPE_VERTICES = _box.vertices.shape[0]

class ModelConfiguration:
    EPOCHS = 100
    SEED = 777

    BATCH_SIZE = 128
    ACCUMULATION_STEPS = 1

    ATTENTION_DIM = 256
    NUM_HEADS = 8
    HIDDEN_DIM = 512

    LR_LATENT_SHAPES = 1e-4
    LR_DECODER = 1e-4

    LATENT_SHAPES_NOISE_RECONSTRUCTION = 0.10

    LOSS_TRAIN_WEIGHT = 0.1
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
    NUM_LAYERS = 10
    NUM_BLOCKS = 2

    CDIST_K = 3
    USE_CDIST = False
    USE_ATTENTION = False

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
