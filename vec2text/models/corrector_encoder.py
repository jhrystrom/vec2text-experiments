from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F  # type: ignore
import transformers

from vec2text.models.config import InversionConfig


@dataclass
class EnhancedOutput:
    loss: torch.Tensor
    logits: torch.Tensor
    centroid_loss: torch.Tensor
    original_loss: torch.Tensor


class CorrectorEncoderModel(transformers.PreTrainedModel):
    """Embeds text and concats with a provided embedding, now with codebook support."""

    config_class = InversionConfig
    encoder_decoder: transformers.PreTrainedModel

    def __init__(
        self,
        config: InversionConfig,
    ):
        super().__init__(config=config)
        if config.embedder_model_api:
            embedder_dim = 1536
        else:
            embedder_dim = 768
        bottleneck_dim = embedder_dim

        num_repeat_tokens = config.num_repeat_tokens
        ignore_hypothesis_embedding = config.corrector_ignore_hypothesis_embedding
        self.use_ff_dropout = False

        encoder_decoder = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            config.model_name_or_path
        )
        self.encoder_decoder = encoder_decoder
        self.embedder_dim = embedder_dim
        self.num_repeat_tokens = num_repeat_tokens
        self.encoder_hidden_dim = self.encoder_decoder.config.hidden_size
        self.embedding_transform_1 = nn.Sequential(
            nn.Linear(self.embedder_dim, bottleneck_dim),
            nn.Dropout(self.encoder_decoder.config.dropout_rate if self.use_ff_dropout else 0.0),
            nn.GELU(),
            nn.Linear(bottleneck_dim, self.encoder_hidden_dim * num_repeat_tokens),
        )
        self.embedding_transform_2 = nn.Sequential(
            nn.Linear(self.embedder_dim, bottleneck_dim),
            nn.Dropout(self.encoder_decoder.config.dropout_rate if self.use_ff_dropout else 0.0),
            nn.GELU(),
            nn.Linear(bottleneck_dim, self.encoder_hidden_dim * num_repeat_tokens),
        )
        self.embedding_transform_3 = nn.Sequential(
            nn.Linear(self.embedder_dim, bottleneck_dim),
            nn.Dropout(self.encoder_decoder.config.dropout_rate if self.use_ff_dropout else 0.0),
            nn.GELU(),
            nn.Linear(bottleneck_dim, self.encoder_hidden_dim * num_repeat_tokens),
        )
        self.ignore_hypothesis_embedding = ignore_hypothesis_embedding
        self.training_embedding_noise_level = 0
        self.use_ln = True
        if self.use_ln:
            self.layernorm = nn.LayerNorm(self.encoder_hidden_dim)

        # Initialize codebook
        # You can adjust the number of centroids based on your needs
        self.num_centroids = getattr(config, "num_centroids", 1024)
        self.codebook = nn.Parameter(torch.randn(self.num_centroids, self.embedder_dim))
        # L2 normalize the codebook vectors
        with torch.no_grad():
            self.codebook.copy_(F.normalize(self.codebook, dim=1))

        # Weight for codebook loss term
        self.codebook_loss_weight = getattr(config, "codebook_loss_weight", 0.1)

    def find_nearest_centroid(self, embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Find the nearest centroid for each embedding.

        Args:
            embeddings: Tensor of shape (batch_size, embedder_dim)

        Returns:
            Tuple of (nearest_centroids, centroid_indices)
                nearest_centroids: Tensor of shape (batch_size, embedder_dim)
                centroid_indices: Tensor of shape (batch_size,)
        """
        # Normalize embeddings and codebook for cosine similarity
        embeddings_normalized = F.normalize(embeddings, dim=1)
        codebook_normalized = F.normalize(self.codebook, dim=1)

        # Compute cosine similarity
        similarity = torch.matmul(embeddings_normalized, codebook_normalized.t())

        # Find the most similar centroid for each embedding
        centroid_indices = similarity.argmax(dim=1)
        nearest_centroids = self.codebook[centroid_indices]

        return nearest_centroids, centroid_indices

    def get_encoder_embedding(
        self,
        embedding: torch.Tensor,
        hypothesis_embedding: torch.Tensor,
        hypothesis_input_ids: torch.Tensor,
        hypothesis_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, D = embedding.shape
        assert embedding.shape == (batch_size, self.embedder_dim)
        assert hypothesis_embedding.shape == (batch_size, self.embedder_dim)

        if (self.training) and (self.training_embedding_noise_level > 0):
            embedding += self.training_embedding_noise_level * torch.randn(
                embedding.shape, device=embedding.device
            )
            hypothesis_embedding += self.training_embedding_noise_level * torch.randn(
                hypothesis_embedding.shape, device=hypothesis_embedding.device
            )

        if self.ignore_hypothesis_embedding:
            # For "No Feedback" ablation
            hypothesis_embedding = embedding

        # Find nearest centroids
        nearest_centroids_e, _ = self.find_nearest_centroid(embedding)
        nearest_centroids_h, _ = self.find_nearest_centroid(hypothesis_embedding)

        # Take the mean of the embedding and the nearest centroid
        embedding = (embedding + nearest_centroids_e) / 2
        hypothesis_embedding = (hypothesis_embedding + nearest_centroids_h) / 2

        diff_embedding = embedding - hypothesis_embedding

        embedding = self.embedding_transform_1(embedding)
        embedding = embedding.reshape((batch_size, self.num_repeat_tokens, self.encoder_hidden_dim))

        diff_embedding = self.embedding_transform_2(diff_embedding)
        diff_embedding = diff_embedding.reshape(
            (batch_size, self.num_repeat_tokens, self.encoder_hidden_dim)
        )

        hypothesis_embedding = self.embedding_transform_3(hypothesis_embedding)
        hypothesis_embedding = hypothesis_embedding.reshape(
            (batch_size, self.num_repeat_tokens, self.encoder_hidden_dim)
        )
        inputs_embeds = self.encoder_decoder.encoder.embed_tokens(hypothesis_input_ids)

        ones = torch.ones((batch_size, 1), dtype=torch.long, device=hypothesis_input_ids.device)
        sep_token = ones * self.encoder_decoder.config.eos_token_id
        sep_token = self.encoder_decoder.encoder.embed_tokens(sep_token)
        inputs_embeds = torch.cat(
            (
                sep_token,
                embedding,
                sep_token,
                hypothesis_embedding,
                sep_token,
                diff_embedding,
                sep_token,
                inputs_embeds,
            ),
            dim=1,
        )
        if self.use_ln:
            inputs_embeds = self.layernorm(inputs_embeds)
        attention_mask = torch.cat(
            (ones.repeat(1, 4 + 3 * self.num_repeat_tokens), hypothesis_attention_mask),
            dim=1,
        )
        return (inputs_embeds, attention_mask)

    def forward(
        self,
        embedding: torch.Tensor,
        hypothesis_embedding: torch.Tensor,
        hypothesis_input_ids: torch.Tensor,
        hypothesis_attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ):
        inputs_embeds, attention_mask = self.get_encoder_embedding(
            embedding=embedding,
            hypothesis_embedding=hypothesis_embedding,
            hypothesis_input_ids=hypothesis_input_ids,
            hypothesis_attention_mask=hypothesis_attention_mask,
        )

        outputs = self.encoder_decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

        # If we're training, add the codebook loss term
        if self.training and labels is not None:
            # Find nearest centroids
            nearest_centroids_e, _ = self.find_nearest_centroid(embedding)
            nearest_centroids_h, _ = self.find_nearest_centroid(hypothesis_embedding)

            # Calculate distances to nearest centroids
            # Using mean squared error as the distance metric
            centroid_loss_e = F.mse_loss(embedding, nearest_centroids_e)
            centroid_loss_h = F.mse_loss(hypothesis_embedding, nearest_centroids_h)
            centroid_loss = (centroid_loss_e + centroid_loss_h) / 2

            # Combine losses
            total_loss = outputs.loss + self.codebook_loss_weight * centroid_loss

            return EnhancedOutput(
                loss=total_loss,
                logits=outputs.logits,
                centroid_loss=centroid_loss,
                original_loss=outputs.loss,
            )

        return outputs
