import copy

import torch
import torch.nn as nn
import transformers
from sentence_transformers import SentenceTransformer

from vec2text.models.config import InversionConfig
from vec2text.models.model_utils import (
    load_embedder_and_tokenizer,
    load_tokenizer,
    mean_pool,
)

# New: Import our vector quantizer module.
from vec2text.models.vector_quantizer import VectorQuantizer


class InversionModel(transformers.PreTrainedModel):
    config_class = InversionConfig
    embedder: nn.Module
    embedder_tokenizer: transformers.PreTrainedTokenizer
    encoder_decoder: transformers.PreTrainedModel
    tokenizer: transformers.PreTrainedTokenizer
    embedding_transform: nn.Module
    bottleneck_dim: int
    embedder_dim: int
    embedder_no_grad: bool
    embedder_fake_with_zeros: bool
    embedding_transform_strategy: str
    use_frozen_embeddings_as_input: bool
    embedded_tokens: torch.Tensor
    embedder_model_api: str | None

    def __init__(self, config: InversionConfig):
        super().__init__(config=config)
        embedder, embedder_tokenizer = load_embedder_and_tokenizer(
            name=config.embedder_model_name, torch_dtype=config.embedder_torch_dtype
        )
        tokenizer = load_tokenizer(
            config.model_name_or_path,
            max_length=config.max_seq_length,
        )
        embedder_model_api = config.embedder_model_api

        if "t5" in config.model_name_or_path:
            encoder_decoder = transformers.T5ForConditionalGeneration.from_pretrained(
                config.model_name_or_path
            )
        else:
            encoder_decoder = transformers.AutoModelForSeq2SeqLM.from_pretrained(
                config.model_name_or_path
            )

        # Freeze the decoder by setting requires_grad=False for all parameters
        for param in encoder_decoder.parameters():
            param.requires_grad = False

        self.embedder = embedder
        self.encoder_decoder = encoder_decoder

        if embedder_model_api:
            assert config.use_frozen_embeddings_as_input, "Must precompute embeddings with API"
            self.embedder_dim = 1536
            bottleneck_dim = 1536
        elif isinstance(self.embedder, SentenceTransformer):
            self.embedder_dim = self.embedder.get_sentence_embedding_dimension()
            bottleneck_dim = self.embedder_dim
        else:
            self.embedder_dim = self.embedder.config.hidden_size
            bottleneck_dim = self.embedder_dim

        self.embedder_no_grad = config.embedder_no_grad
        self.use_frozen_embeddings_as_input = config.use_frozen_embeddings_as_input
        self.bottleneck_dim = bottleneck_dim

        # Create a projection to map the embedder output into the decoder's hidden space.
        self.embedding_transform = nn.Sequential(
            nn.Linear(self.embedder_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, encoder_decoder.config.hidden_size),
        )

        # Freeze the embedding transform as well
        for param in self.embedding_transform.parameters():
            param.requires_grad = False

        self.tokenizer = tokenizer
        self.embedder_tokenizer = embedder_tokenizer
        self.embedder_model_api = embedder_model_api
        self.embedder_fake_with_zeros = config.embedder_fake_with_zeros
        self.embedding_transform_strategy = "repeat"
        self.noise_level = 0
        self.embeddings_from_layer_n = None

        # --- VQ-VAE components --- #
        self.use_vq: bool = getattr(config, "use_vq", False)
        if self.use_vq:
            print("Using VQ!")
            num_codebook_vectors = getattr(config, "num_codebook_vectors", 499)
            vq_commitment_cost = getattr(config, "vq_commitment_cost", 0.25)
            # We assume the dimension to quantize is that of the decoder hidden size.
            self.vector_quantizer = VectorQuantizer(
                num_codebook_vectors,
                encoder_decoder.config.hidden_size,
                vq_commitment_cost,
            )
            # The vector quantizer is the only component that should be trainable
            # Ensure its parameters require gradients
            for param in self.vector_quantizer.parameters():
                param.requires_grad = True

            self.vq_loss_weight: float = getattr(config, "vq_loss_weight", 1.0)
        else:
            self.vector_quantizer = None
        # --- End VQ-VAE components --- #

        # Freeze the embedder if specified
        if self.embedder_no_grad:
            for param in self.embedder.parameters():
                param.requires_grad = False

    def call_embedding_model(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.embedder(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(outputs, "pooler_output") and (outputs.pooler_output is not None):
            return outputs.pooler_output
        else:
            return mean_pool(outputs.last_hidden_state, attention_mask)

    def embed_and_project(
        self,
        embedder_input_ids: torch.Tensor | None,
        embedder_attention_mask: torch.Tensor | None,
        frozen_embeddings: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert not (
            (embedder_input_ids is None) and (frozen_embeddings is None)
        ), "Need either input_ids or precomputed embeddings"
        if frozen_embeddings is not None:
            embeddings = frozen_embeddings
            assert len(embeddings.shape) == 2
        elif self.embedder_no_grad:
            with torch.no_grad():
                embeddings = self.call_embedding_model(
                    input_ids=embedder_input_ids, attention_mask=embedder_attention_mask
                )
        else:
            embeddings = self.call_embedding_model(
                input_ids=embedder_input_ids, attention_mask=embedder_attention_mask
            )

        # Project embeddings to decoder hidden space (with gradient detached to ensure only VQ updates)
        with torch.no_grad():
            projected = self.embedding_transform(embeddings)  # shape: (B, hidden_size)

        # Apply VQ if enabled (during both training and generation)
        if self.use_vq and self.vector_quantizer is not None:
            # The only place where gradients are tracked
            quantized, _ = self.vector_quantizer(projected.detach())
            projected = quantized

        # Reshape to sequence form (B, 1, hidden_size)
        projected = projected.reshape((projected.shape[0], 1, -1))
        attention_mask_out = torch.ones(
            (projected.shape[0], projected.shape[1]), device=projected.device
        )
        return projected, attention_mask_out

    def forward(
        self,
        embedder_input_ids: torch.Tensor,
        embedder_attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        frozen_embeddings: torch.Tensor | None = None,
        decoder_input_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        # Compute projected hypothesis embeddings from the embedder.
        inputs_embeds, attention_mask = self.embed_and_project(
            embedder_input_ids=embedder_input_ids,
            embedder_attention_mask=embedder_attention_mask,
            frozen_embeddings=frozen_embeddings,
        )

        # Calculate VQ loss if using VQ
        vq_loss = 0.0
        if self.use_vq and self.vector_quantizer is not None:
            # We need to calculate the VQ loss separately
            if frozen_embeddings is not None:
                embeddings = frozen_embeddings
            elif self.embedder_no_grad:
                with torch.no_grad():
                    embeddings = self.call_embedding_model(
                        input_ids=embedder_input_ids, attention_mask=embedder_attention_mask
                    )
            else:
                embeddings = self.call_embedding_model(
                    input_ids=embedder_input_ids, attention_mask=embedder_attention_mask
                )
            # Calculate the VQ loss - this is where the codebook will be updated
            _, vq_loss = self.vector_quantizer(embeddings.detach())

        # Forward pass through the encoder-decoder model (with no_grad to ensure decoder isn't updated)
        with torch.no_grad():
            outputs = self.encoder_decoder(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                decoder_input_ids=decoder_input_ids,
            )

        # For backpropagation, we need to create a new loss that only depends on the VQ component
        if self.use_vq and self.vector_quantizer is not None and vq_loss > 0:
            # Create a new outputs object if necessary, or modify the existing one
            # We don't use the original model's loss, only the VQ loss for optimization
            outputs.loss = self.vq_loss_weight * vq_loss

        return outputs

    def generate(
        self,
        inputs: dict[str, torch.Tensor],
        generation_kwargs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        generation_kwargs = copy.copy(generation_kwargs)
        inputs_embeds, attention_mask = self.embed_and_project(
            embedder_input_ids=inputs.get("embedder_input_ids"),
            embedder_attention_mask=inputs.get("embedder_attention_mask"),
            frozen_embeddings=inputs.get("frozen_embeddings"),
        )
        # For generation, we use the frozen decoder
        with torch.no_grad():
            return self.encoder_decoder.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                **generation_kwargs,
            )
