if __name__ == "__main__":
    # CUDA_VISIBLE_DEVICE=0 python train.py

    import os
    import sys

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    import torch

    if os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")) not in sys.path:
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

    from latent_shape_interpolator.src.config import Configuration
    from latent_shape_interpolator.src.data import SDFDataset
    from latent_shape_interpolator.src.model import SDFDecoder, LatentShapes
    from latent_shape_interpolator.src.trainer import Trainer

    configuration = Configuration()
    configuration.set_seed()

    sdf_dataset = SDFDataset.create_dataset(
        data_dir=configuration.DATA_PATH_PROCESSED, configuration=configuration, data_slicer=10
    )

    latent_shapes = LatentShapes(latent_shapes=sdf_dataset.latent_shapes, noise_min=-0.1, noise_max=0.1)

    sdf_decoder = SDFDecoder(configuration=configuration)

    states = torch.load(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../runs/07-09-2025__09-44-54/states.pth"))
    )

    sdf_decoder.load_state_dict(states["state_dict_model"])
    latent_shapes.load_state_dict(states["state_dict_latent_shapes"])

    # latent_shapes_batch = sdf_dataset.latent_shapes[
    #     torch.randperm(sdf_dataset.num_classes)[: configuration.RECONSTRUCTION_COUNT]
    # ]

    latent_shapes_batch = latent_shapes(torch.randperm(sdf_dataset.num_classes)[: configuration.RECONSTRUCTION_COUNT])

    for _ in range(1):
        reconstruction_results = sdf_decoder.reconstruct(
            latent_shapes_batch,
            save_path="",
            normalize=True,
            check_watertight=False,
            add_noise=False,
            rescale=True,
        )
