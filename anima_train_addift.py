"""Dedicated entry point for Anima ADDifT paired-image LoRA training."""

from library import train_util
from library.device_utils import init_ipex

init_ipex()

import anima_train_network


def main() -> None:
    parser = anima_train_network.setup_parser()
    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    if args.dataset_config is None:
        raise ValueError("--dataset_config is required")

    # This entry point is intentionally ADDifT-only.
    args.paired_difference_mode = True
    args.network_train_unet_only = True
    args.cache_latents = False
    args.cache_latents_to_disk = False
    args.gradient_checkpointing = True
    args.max_data_loader_n_workers = 0
    args.persistent_data_loader_workers = False

    trainer = anima_train_network.AnimaNetworkTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
