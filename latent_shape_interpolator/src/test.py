
import os
import sys

# print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch


if os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from latent_shape_interpolator.src.config import Configuration
from latent_shape_interpolator.src.data import SDFDataset
from latent_shape_interpolator.src.model import SDFDecoder
from latent_shape_interpolator.src.trainer import Trainer

configuration = Configuration()
configuration.set_seed()

sdf_dataset = SDFDataset.create_dataset(
    data_dir=configuration.DATA_PATH_PROCESSED, configuration=configuration, data_slicer=10
)

sdf_decoder = SDFDecoder(
    latent_shapes=sdf_dataset.latent_shapes,
    configuration=configuration,
)

sdf_decoder_module = sdf_decoder

if configuration.USE_MULTI_GPUS:
    sdf_decoder = torch.nn.DataParallel(sdf_decoder, device_ids=[i for i in range(torch.cuda.device_count())])

    sdf_decoder_module = sdf_decoder.module

sdf_decoder_module.load_state_dict(
    torch.load(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../runs/07-03-2025__09-27-28/states.pth"))
    )["state_dict_model"]
)


latent_shapes_batch = sdf_decoder_module.latent_shapes_embedding(
    torch.randperm(sdf_dataset.num_classes)[: configuration.RECONSTRUCTION_COUNT]
)

reconstruction_results = sdf_decoder.reconstruct(
    sdf_dataset.latent_shapes[0].unsqueeze(0),
    save_path="",
    normalize=True,
    check_watertight=False,
    add_noise=False,
    rescale=True,
)