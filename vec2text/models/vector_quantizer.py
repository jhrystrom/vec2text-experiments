from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F  # type: ignore

from vec2text.utils.init_codebook import initialize_codebook
class VectorQuantizer(nn.Module):
    def __init__(
        self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25, 
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
        # Normalize inputs and codebook for cosine similarity
        inputs_normalized = F.normalize(inputs, p=2, dim=1)
        codebook_normalized = F.normalize(self.codebook, p=2, dim=1)

        # Compute cosine similarity between each input and codebook vector
        # Cosine similarity = dot product of normalized vectors
        # We use negative similarity since we want to find the closest (most similar) vector
        # inputs_normalized: (B, D); codebook_normalized: (K, D)
        cosine_similarities = torch.matmul(
            inputs_normalized, codebook_normalized.t()
        )  # shape: (B, K)
        cosine_distances = -cosine_similarities  # Convert to distance (negative similarity)

        # Find the nearest codebook vector for each input
        encoding_indices = torch.argmin(cosine_distances, dim=1)  # shape: (B,)
        quantized = self.codebook[encoding_indices]  # (B, D)

        # Compute the VQ loss (with a commitment loss term)
        loss = F.mse_loss(quantized.detach(), inputs) + self.commitment_cost * F.mse_loss(
            quantized, inputs.detach()
        )
        # Use the straight-through estimator
        quantized = inputs + (quantized - inputs).detach()
        return quantized, loss
