import torch.nn as nn
from transformers import AutoModel


class ImageEncoder(nn.Module):
    def __init__(
        self,
        output_dim=512,
        img_model="facebook/dinov3-vitb16-pretrain-lvd1689m",
        unfreeze_n_blocks=4,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(img_model)

        # freeze all parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

        hidden_size = getattr(self.encoder, "embed_dim", None)
        if hidden_size is None:
            hidden_size = getattr(self.encoder.config, "hidden_size", None)
        if hidden_size is None:
            raise AttributeError("Unable to determine encoder hidden size for projection.")
        proj_hidden = hidden_size * 2
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, proj_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(proj_hidden, output_dim),
        )

    @staticmethod
    def _get_transformer_blocks(model):
        if hasattr(model, "blocks"):
            return model.blocks
        if hasattr(model, "encoder"):
            encoder = model.encoder
            if hasattr(encoder, "layer"):
                return encoder.layer
            if hasattr(encoder, "layers"):
                return encoder.layers
            if hasattr(encoder, "blocks"):
                return encoder.blocks
        if hasattr(model, "layer"):
            return model.layer
        if hasattr(model, "layers"):
            return model.layers
        raise AttributeError("Unable to find transformer blocks to unfreeze.")

    @staticmethod
    def _get_norm_layer(model):
        if hasattr(model, "norm"):
            return model.norm
        if hasattr(model, "layernorm"):
            return model.layernorm
        if hasattr(model, "encoder"):
            encoder = model.encoder
            if hasattr(encoder, "norm"):
                return encoder.norm
            if hasattr(encoder, "layernorm"):
                return encoder.layernorm
        raise AttributeError("Unable to find normalization layer to unfreeze.")

    def forward(self, x):
        if hasattr(self.encoder, "forward_features"):
            dino_output = self.encoder.forward_features(x)
            if isinstance(dino_output, dict) and "x_norm_clstoken" in dino_output:
                x = dino_output["x_norm_clstoken"]
            elif isinstance(dino_output, dict) and "x_prenorm" in dino_output:
                x = dino_output["x_prenorm"][:, 0]
            else:
                x = dino_output
        else:
            output = self.encoder(x)
            if hasattr(output, "pooler_output") and output.pooler_output is not None:
                x = output.pooler_output
            elif hasattr(output, "last_hidden_state"):
                x = output.last_hidden_state[:, 0]
            else:
                raise AttributeError("Encoder output missing pooler or hidden states.")
        x = self.proj(x)
        return x
