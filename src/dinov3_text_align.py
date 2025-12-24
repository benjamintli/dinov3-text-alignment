import torch
import torch.nn.functional as F
import faiss
import numpy as np
from src.loss import ContrastiveLoss
from src.image_encoder import ImageEncoder
from src.text_encoder import TextEncoder


class DinoV3TextAlignment(torch.nn.Module):
    def __init__(
        self,
        txt_model="sentence-transformers/all-MiniLM-L6-v2",
        img_model="facebook/dinov3-vitb16-pretrain-lvd1689m",
        embed_size=512,  # output dimension of the encoder
        unfreeze_n_blocks=4,
        lr=0.0001,
        warmup_epochs=0,
        weight_decay=0.0001,
        milestones=[5, 10, 15],
        lr_mult=0.1,
    ):
        super().__init__()

        self.txt_model = txt_model
        self.img_model = img_model
        self.embed_size = embed_size
        self.unfreeze_n_blocks = unfreeze_n_blocks
        self.lr = lr
        self.warmup_epochs = warmup_epochs
        self.weight_decay = weight_decay
        self.milestones = milestones
        self.lr_mult = lr_mult

        self.img_encoder = ImageEncoder(
            self.embed_size, self.img_model, unfreeze_n_blocks
        )
        self.txt_encoder = TextEncoder(
            self.embed_size, self.txt_model, unfreeze_n_blocks
        )
        self.loss_fn = ContrastiveLoss(temperature=0.07)

    def forward(self, image, captions, masks):
        """
        Define the forward pass of the pipeline.
        """
        # compute image embeddings
        image_embedding = self.img_encoder(image)  # (batch_size, out_dim)
        image_embedding = F.normalize(
            image_embedding, p=2, dim=-1
        )  # normalize embeddings

        # compute text embeddings
        text_embedding = self.txt_encoder(
            captions, masks
        )  # (batch_size, nb_captions, out_dim)
        text_embedding = F.normalize(
            text_embedding, p=2, dim=-1
        )  # normalize embeddings

        return image_embedding, text_embedding

    @staticmethod
    def _calculate_recall(
        img_descriptors, txt_descriptors, labels, k_values=[1, 5, 10]
    ):
        """
        Calculate the recall at k for the given img_descriptors as gallery
        and txt_descriptors as queries.
        """
        embed_size = img_descriptors.shape[1]
        faiss_index = faiss.IndexFlatL2(embed_size)

        faiss_index.add(img_descriptors)  # add images to the index
        _, predictions = faiss_index.search(
            txt_descriptors, max(k_values)
        )  # search for the top k images for each text query

        correct_at_k = np.zeros(len(k_values))
        for q_idx, pred in enumerate(predictions):
            for i, n in enumerate(k_values):
                # if in top N then also in top NN, where NN > N
                if np.any(np.in1d(pred[:n], labels[q_idx])):
                    correct_at_k[i:] += 1
                    break

        correct_at_k /= len(labels)

        return correct_at_k
