import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.append(str(Path(__file__).parent.parent.parent))

from fastspeech2.data import FastSpeech2Collate, LJSpeechDataset
from fastspeech2.training import Trainer


def main():
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument(
        "--resume", type=str, default=None, help="Checkpoint path to resume"
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    config = OmegaConf.to_container(config, resolve=True)

    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config["seed"])

    preprocessed_dir = config["paths"]["preprocessed_dir"]

    train_dataset = LJSpeechDataset(preprocessed_dir, split="train")
    val_dataset = LJSpeechDataset(preprocessed_dir, split="val")

    train_dataset.collate_fn = FastSpeech2Collate()
    val_dataset.collate_fn = FastSpeech2Collate()

    trainer = Trainer(config, train_dataset, val_dataset)

    if args.resume:
        from fastspeech2.utils import load_checkpoint

        checkpoint = load_checkpoint(args.resume, device=config["device"])
        trainer.model.load_state_dict(checkpoint["model"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer"])
        trainer.current_epoch = checkpoint["epoch"]
        trainer.global_step = checkpoint["global_step"]
        trainer.global_step = checkpoint['global_step']
        print(f"Resumed from epoch {trainer.current_epoch}, step {trainer.global_step}")

    trainer.train()


if __name__ == "__main__":
    main()
