import torch
import torch.nn as nn

from .decoder import get_decoder
from .encoder import get_encoder

from configs import Config


class TrOMR(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.encoder = get_encoder(config)
        self.decoder = get_decoder(config)
        self.config = config

    def forward(self, inputs, rhythms_seq, pitchs_seq, 
                lifts_seq, note_seq, mask, **kwargs):
        mask = None
        encoded = self.encoder(inputs)
        loss = self.decoder(rhythms_seq, pitchs_seq, 
                            lifts_seq, note_seq, 
                            context=encoded, mask=mask, **kwargs)
        return loss

    @torch.no_grad()
    def generate(self, x: torch.Tensor, temperature: float = 0.25):
        start_token = (
                torch.LongTensor([self.config.bos_token]*len(x))[:, None]
            ).to(x.device)
        nonote_token = (
                torch.LongTensor([self.config.nonote_token]*len(x))[:, None]
            ).to(x.device)

        out_lift, out_pitch, out_rhythm = self.decoder.generate(
            start_token, nonote_token, self.config.max_seq_len,
            eos_token=self.config.eos_token, context=self.encoder(x), 
            temperature=temperature)
        
        return out_lift, out_pitch, out_rhythm
