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

    _sdf_decoder_optimizer = getattr(torch.optim, configuration.OPTIMIZER)
    sdf_decoder_optimizer = _sdf_decoder_optimizer(
        [
            {"params": sdf_decoder_module.latent_shapes_embedding.parameters(), "lr": configuration.LR_LATENT_SHAPES},
            {"params": sdf_decoder_module.xyz_projection.parameters(), "lr": configuration.LR_DECODER},
            {"params": sdf_decoder_module.latent_projection.parameters(), "lr": configuration.LR_DECODER},
            {"params": sdf_decoder_module.attention.parameters(), "lr": configuration.LR_DECODER},
            {"params": sdf_decoder_module.ff.parameters(), "lr": configuration.LR_DECODER},
            {"params": sdf_decoder_module.layer_norm_1.parameters(), "lr": configuration.LR_DECODER},
            {"params": sdf_decoder_module.layer_norm_2.parameters(), "lr": configuration.LR_DECODER},
            *[{"params": block.parameters(), "lr": configuration.LR_DECODER} for block in sdf_decoder_module.blocks],
        ]
    )

    sdf_decoder_trainer = Trainer(
        sdf_decoder=sdf_decoder,
        sdf_decoder_optimizer=sdf_decoder_optimizer,
        sdf_dataset=sdf_dataset,
        configuration=configuration,
        # pretrained_dir="latent_shape_interpolator/runs/06-23-2025__21-24-31",
    )

    sdf_decoder_trainer.train()
