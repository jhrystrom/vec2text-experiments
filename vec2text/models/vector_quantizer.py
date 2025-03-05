# vec2text/models/vector_quantizer.py
import torch
import torch.nn as nn
import torch.nn.functional as F  # type: ignore


class VectorQuantizer(nn.Module):
    def __init__(
        self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25
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
        # Compute L2 distance between each input and codebook vector.
        # inputs: (B, D); codebook: (K, D)
        distances = (
            torch.sum(inputs**2, dim=1, keepdim=True)
            + torch.sum(self.codebook**2, dim=1)
            - 2 * torch.matmul(inputs, self.codebook.t())
        )  # shape: (B, K)
        # Find the nearest codebook vector for each input.
        encoding_indices = torch.argmin(distances, dim=1)  # shape: (B,)
        quantized = self.codebook[encoding_indices]  # (B, D)

        # Compute the VQ loss (with a commitment loss term).
        loss = F.mse_loss(quantized.detach(), inputs) + self.commitment_cost * F.mse_loss(
            quantized, inputs.detach()
        )
        # Use the straight-through estimator.
        quantized = inputs + (quantized - inputs).detach()
        return quantized, loss
