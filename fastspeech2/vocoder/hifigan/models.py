import torch
import torch.nn as nn
from transformers import AutoModel


class HiFiGAN:
    def __init__(self, checkpoint_path=None, device="cuda"):
        self.device = device

        if checkpoint_path:
            self.model = torch.load(checkpoint_path, map_location=device)
        else:
            self.model = AutoModel.from_pretrained(
                "facebook/hifigan", trust_remote_code=True
            )

        self.model = self.model.to(device)
        self.model.eval()

    def inference(self, mel):
        if isinstance(mel, torch.Tensor):
            mel = mel.cpu().numpy()

        mel = torch.FloatTensor(mel).unsqueeze(0).to(self.device)

        with torch.no_grad():
            wav = self.model(mel)

        return wav.squeeze().cpu().numpy()
