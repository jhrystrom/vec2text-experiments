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
        self.tokenizer = tokenizer
        self.embedder_tokenizer = embedder_tokenizer
        self.embedder_model_api = embedder_model_api
        self.embedder_fake_with_zeros = config.embedder_fake_with_zeros
        self.embedding_transform_strategy = "repeat"
        self.noise_level = 0
        self.embeddings_from_layer_n = None

        # --- New: VQ-VAE components --- #
        self.use_vq: bool = getattr(config, "use_vq", False)
        if self.use_vq:
            num_codebook_vectors = getattr(config, "num_codebook_vectors", 512)
            vq_commitment_cost = getattr(config, "vq_commitment_cost", 0.25)
            # We assume the dimension to quantize is that of the decoder hidden size.
            self.vector_quantizer = VectorQuantizer(
                num_codebook_vectors,
                encoder_decoder.config.hidden_size,
                vq_commitment_cost,
            )
            self.vq_loss_weight: float = getattr(config, "vq_loss_weight", 1.0)
        else:
            self.vector_quantizer = None
        # --- End VQ-VAE components --- #

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

        # Project embeddings to decoder hidden space.
        projected = self.embedding_transform(embeddings)  # shape: (B, hidden_size)
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
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor | None = None,
        frozen_embeddings: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        # If labels provided, adjust decoder inputs (shift right) accordingly.
        if labels is not None:
            input_ids = input_ids[:, :-1]
            attention_mask = attention_mask[:, :-1]

        # Compute projected hypothesis embeddings from the embedder.
        embed_inputs_embeds, embed_attention_mask = self.embed_and_project(
            embedder_input_ids=embedder_input_ids,
            embedder_attention_mask=embedder_attention_mask,
            frozen_embeddings=frozen_embeddings,
        )

        # --- VQ-VAE modification --- #
        vq_loss = 0.0
        if self.use_vq and self.vector_quantizer is not None:
            # Squeeze the sequence dimension (currently 1) to get (B, hidden_size)
            hypothesis_embedding = embed_inputs_embeds.squeeze(1)
            quantized, vq_loss = self.vector_quantizer(hypothesis_embedding)
            # Replace hypothesis with quantized version and reshape back to sequence form.
            embed_inputs_embeds = quantized.unsqueeze(1)
        # --- End VQ-VAE modification --- #

        # Get the decoder's input embeddings
        input_embeddings_table = self.encoder_decoder.get_input_embeddings()
        decoder_embeddings = input_embeddings_table(input_ids)

        # Concatenate the (possibly quantized) hypothesis embedding with decoder embeddings
        inputs_embeds = torch.cat((embed_inputs_embeds, decoder_embeddings), dim=1)
        full_attention_mask = torch.cat((embed_attention_mask, attention_mask), dim=1)

        outputs = self.encoder_decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            labels=labels,
        )

        # Combine the original reconstruction loss with the VQ loss (if any)
        loss = outputs.loss
        if self.use_vq and self.vector_quantizer is not None:
            loss = loss + self.vq_loss_weight * vq_loss
        outputs.loss = loss
        return outputs

    def generate(
        self,
        inputs: dict[str, torch.Tensor],
        generation_kwargs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        generation_kwargs = copy.copy(generation_kwargs)
        inputs_embeds, attention_mask = self.embed_and_project(
            embedder_input_ids=inputs["embedder_input_ids"],
            embedder_attention_mask=inputs["embedder_attention_mask"],
            frozen_embeddings=inputs.get("frozen_embeddings"),
        )
        # For generation we do not quantize (or you could apply VQ if desired)
        return self.encoder_decoder.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generation_kwargs,
        )
