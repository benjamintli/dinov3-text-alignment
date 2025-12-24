import torch.nn as nn
from transformers import AutoModel


class TextEncoder(nn.Module):
    def __init__(
        self,
        output_dim=512,
        lang_model="sentence-transformers/all-MiniLM-L6-v2",
        unfreeze_n_blocks=4,
    ):
        super().__init__()
        self.lang_model = lang_model
        self.encoder = AutoModel.from_pretrained(self.lang_model)

        for param in self.encoder.parameters():
            param.requires_grad = False

        for layer in self.encoder.encoder.layer[-unfreeze_n_blocks:]:
            for param in layer.parameters():
                param.requires_grad = True

        # unfreeze the pooler layer
        for param in self.encoder.pooler.parameters():
            param.requires_grad = True

        proj_hidden = self.encoder.config.hidden_size * 2
        self.proj = nn.Sequential(
            nn.LayerNorm(self.encoder.config.hidden_size),
            nn.Linear(self.encoder.config.hidden_size, proj_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(proj_hidden, output_dim),
        )

    def forward(self, input_ids, attention_mask=None):
        x = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state[:, 0]
        x = self.proj(x)
        return x
