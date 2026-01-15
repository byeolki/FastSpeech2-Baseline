import logging
import sys
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, log_dir: str, name: str = "fastspeech2"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            file_handler = logging.FileHandler(self.log_dir / "train.log")
            file_handler.setLevel(logging.INFO)

            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)

            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

        self.writer = SummaryWriter(log_dir=str(self.log_dir))

    def info(self, msg: str):
        self.logger.info(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def error(self, msg: str):
        self.logger.error(msg)

    def log_scalar(self, tag: str, value: float, step: int):
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: dict, step: int):
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_audio(self, tag: str, audio: any, step: int, sample_rate: int):
        self.writer.add_audio(tag, audio, step, sample_rate=sample_rate)

    def log_image(self, tag: str, image: any, step: int):
