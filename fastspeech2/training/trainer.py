from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..losses import FastSpeech2Loss
from ..models.fastspeech2 import FastSpeech2
from ..utils import Logger, load_checkpoint, save_checkpoint
from .optimizer import get_optimizer
from .scheduler import get_scheduler


class Trainer:
    def __init__(self, config, train_dataset, val_dataset):
        self.device = torch.device(config["device"])

        self.model = FastSpeech2(config).to(self.device)
        self.criterion = FastSpeech2Loss(config)
        self.optimizer = get_optimizer(self.model, config)
        self.scheduler = get_scheduler(self.optimizer, config)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config["train"]["batch_size"],
            shuffle=True,
            num_workers=config["train"]["num_workers"],
            pin_memory=config["train"]["pin_memory"],
            collate_fn=train_dataset.collate_fn
            if hasattr(train_dataset, "collate_fn")
            else None,
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config["train"]["batch_size"],
            shuffle=False,
            num_workers=config["train"]["num_workers"],
            pin_memory=config["train"]["pin_memory"],
            collate_fn=val_dataset.collate_fn
            if hasattr(val_dataset, "collate_fn")
            else None,
        )

        self.logger = Logger(config)
        self.scaler = (
            torch.amp.GradScaler("cuda") if config["train"]["mixed_precision"] else None
        )

        self.global_step = 0
        self.current_epoch = 0

    def train_epoch(self):
        self.model.train()
        total_loss = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")

        for batch_idx, batch in enumerate(pbar):
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            self.optimizer.zero_grad()

            if self.scaler:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    predictions = self.model(
                        text=batch["texts"],
                        src_mask=batch["src_masks"],
                        mel_mask=batch["mel_masks"],
                        duration_target=batch["durations"],
                        pitch_target=batch["pitches"],
                        energy_target=batch["energies"],
                        max_len=batch["mels"].size(1),
                    )

                    losses = self.criterion(
                        predictions,
                        batch,
                        {
                            "src_masks": batch["src_masks"],
                            "mel_masks": batch["mel_masks"],
                        },
                    )

                self.scaler.scale(losses["total"]).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config["train"]["gradient_clip_val"]
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                predictions = self.model(
                    text=batch["texts"],
                    src_mask=batch["src_masks"],
                    mel_mask=batch["mel_masks"],
                    duration_target=batch["durations"],
                    pitch_target=batch["pitches"],
                    energy_target=batch["energies"],
                    max_len=batch["mels"].size(1),
                )

                losses = self.criterion(
                    predictions,
                    batch,
                    {"src_masks": batch["src_masks"], "mel_masks": batch["mel_masks"]},
                )

                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config["train"]["gradient_clip_val"]
                )
                self.optimizer.step()

            self.scheduler.step()

            total_loss += losses["total"].item()

            if self.global_step % self.config["train"]["log_every_n_steps"] == 0:
                metrics = {
                    "train/loss": losses["total"].item(),
                    "train/mel_loss": losses["mel"].item(),
                    "train/postnet_loss": losses["postnet"].item(),
                    "train/duration_loss": losses["duration"].item(),
                    "train/pitch_loss": losses["pitch"].item(),
                    "train/energy_loss": losses["energy"].item(),
                    "train/lr": self.optimizer.param_groups[0]["lr"],
                }
                self.logger.log(metrics, self.global_step)

            pbar.set_postfix({"loss": losses["total"].item()})
            self.global_step += 1

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0

        for batch in tqdm(self.val_loader, desc="Validation"):
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            predictions = self.model(
                text=batch["texts"],
                src_mask=batch["src_masks"],
                mel_mask=batch["mel_masks"],
                duration_target=batch["durations"],
                pitch_target=batch["pitches"],
                energy_target=batch["energies"],
                max_len=batch["mels"].size(1),
            )

            losses = self.criterion(
                predictions,
                batch,
                {"src_masks": batch["src_masks"], "mel_masks": batch["mel_masks"]},
            )

            total_loss += losses["total"].item()

        avg_loss = total_loss / len(self.val_loader)

        metrics = {"val/loss": avg_loss}
        self.logger.log(metrics, self.global_step)

        return avg_loss

    def train(self):
        for epoch in range(self.current_epoch, self.config["train"]["epochs"]):
            self.current_epoch = epoch

            train_loss = self.train_epoch()

            if epoch % self.config["train"]["validate_every_n_epochs"] == 0:
                val_loss = self.validate()
                print(
                    f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}"
                )

            if epoch % self.config["train"]["save_every_n_epochs"] == 0:
                state = {
                    "epoch": epoch,
                    "global_step": self.global_step,
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "config": self.config,
                }
                save_checkpoint(
                    state, self.config["paths"]["checkpoint_dir"], self.global_step
                    self.global_step
                )

        self.logger.finish()
