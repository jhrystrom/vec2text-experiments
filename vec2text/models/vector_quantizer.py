import torch
import torch.nn as nn
import torch.nn.functional as F  # type: ignore
from torchmetrics.functional.pairwise import pairwise_cosine_similarity


class VectorQuantizer(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
    ) -> None:
        """
        Args:
            num_embeddings: number of codebook vectors.
            embedding_dim: dimensionality of each codebook vector.
            commitment_cost: weight for the commitment loss.
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        # Initialize codebook randomly; later you can reinitialize with k-means.
        self.codebook = nn.Parameter(torch.randn(num_embeddings, embedding_dim))

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            inputs: tensor of shape (batch, embedding_dim)
        Returns:
            quantized: the quantized embeddings (same shape as inputs)
            loss: the quantization loss to add to the overall loss.
        """

        cosine_similarities = pairwise_cosine_similarity(inputs, self.codebook)
        cosine_distances = -cosine_similarities  # Convert to distance (negative similarity)

        encoding_indices = torch.argmin(cosine_distances, dim=1)  # shape: (B,)
        # Problem: Why all the same value (7)?
        print(f"{encoding_indices=}")
        quantized = self.codebook[encoding_indices]  # (B, D)

        # Don't use detach here! (for quantized)
        # Compute the VQ loss (with a commitment loss term)
        loss = F.mse_loss(quantized.detach(), inputs) + self.commitment_cost * F.mse_loss(
            quantized, inputs.detach()
        )

        # Don't use detach here! (for quantized) - or detach both input
        # Use the straight-through estimator
        quantized = inputs + (quantized - inputs)  # .detach()
        return quantized, loss
