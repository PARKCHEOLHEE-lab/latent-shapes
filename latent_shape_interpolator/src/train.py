if __name__ == "__main__":
    # CUDA_VISIBLE_DEVICE=0 python train.py

    import os
    import sys
    import torch

    print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")

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

    latent_shapes = LatentShapes(
        latent_shapes=sdf_dataset.latent_shapes,
        noise_min=-configuration.LATENT_SHAPES_NOISE_RECONSTRUCTION,
        noise_max=configuration.LATENT_SHAPES_NOISE_RECONSTRUCTION,
    )

    sdf_decoder = SDFDecoder(configuration=configuration)

    sdf_decoder_module = sdf_decoder

    if configuration.USE_MULTI_GPUS and torch.cuda.device_count() >= 2:
        sdf_decoder = torch.nn.DataParallel(sdf_decoder, device_ids=[i for i in range(torch.cuda.device_count())])

        sdf_decoder_module = sdf_decoder.module

    _sdf_decoder_optimizer = getattr(torch.optim, configuration.OPTIMIZER)
    sdf_decoder_optimizer = _sdf_decoder_optimizer(sdf_decoder_module.parameters(), lr=configuration.LR_DECODER)

    _latent_shapes_optimizer = getattr(torch.optim, configuration.OPTIMIZER)
    latent_shapes_optimizer = _latent_shapes_optimizer(latent_shapes.parameters(), lr=configuration.LR_LATENT_SHAPES)

    sdf_decoder_trainer = Trainer(
        latent_shapes=latent_shapes,
        latent_shapes_optimizer=latent_shapes_optimizer,
        sdf_decoder=sdf_decoder,
        sdf_decoder_optimizer=sdf_decoder_optimizer,
        sdf_dataset=sdf_dataset,
        configuration=configuration,
        # pretrained_dir="latent_shape_interpolator/runs/07-05-2025__15-08-21",
    )

    sdf_decoder_trainer.train()
