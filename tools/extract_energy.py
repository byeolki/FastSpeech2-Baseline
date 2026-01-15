import argparse
import json
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from fastspeech2.utils import AudioProcessor


def extract_energy(config):
    raw_dir = Path(config["paths"]["raw_dir"])
    preprocessed_dir = Path(config["paths"]["preprocessed_dir"])
    energy_dir = preprocessed_dir / "energy"
    energy_dir.mkdir(parents=True, exist_ok=True)

    audio_processor = AudioProcessor(config["audio"])

    with open(preprocessed_dir / "train.json", "r") as f:
        train_metadata = json.load(f)
    with open(preprocessed_dir / "val.json", "r") as f:
        val_metadata = json.load(f)

    metadata = train_metadata + val_metadata

    for item in tqdm(metadata, desc="Extracting energy"):
        basename = item["basename"]
        wav_path = raw_dir / "wavs" / f"{basename}.wav"

        try:
            wav = audio_processor.load_wav(str(wav_path))
            energy = audio_processor.extract_energy(
                wav, use_log=config["audio"]["energy"]["use_log_scale"]
            )

            np.save(energy_dir / f"{basename}.npy", energy)

        except Exception as e:
            print(f"Error processing {basename}: {e}")
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    config = OmegaConf.to_container(config, resolve=True)

    extract_energy(config)
