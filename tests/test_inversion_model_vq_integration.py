import pytest
import torch

from vec2text.models.config import InversionConfig
from vec2text.models.inversion import InversionModel
from vec2text.models.model_utils import device


def get_config(use_vq: bool) -> InversionConfig:
    config_dict = {
        "model_name_or_path": "t5-small",  # Use a small pretrained T5 model.
        "embedder_model_name": "bert",  # Loads from "bert-base-uncased" (hidden size 768)
        "max_seq_length": 16,
        "num_repeat_tokens": 2,
        "use_vq": use_vq,  # Enable or disable vector quantization.
        "num_codebook_vectors": 16,
        "vq_commitment_cost": 0.25,
        "vq_loss_weight": 1.0,
        "embedder_torch_dtype": "float32",
        "embedder_model_api": None,
        "embedder_no_grad": True,
        "use_frozen_embeddings_as_input": True,
        "embedder_fake_with_zeros": False,
    }
    return InversionConfig(**config_dict)


@pytest.fixture(scope="module")
def model_with_vq() -> InversionModel:
    config = get_config(use_vq=True)
    model = InversionModel(config)
    model.to(device)
    model.eval()
    return model


@pytest.fixture(scope="module")
def model_without_vq() -> InversionModel:
    config = get_config(use_vq=False)
    model = InversionModel(config)
    model.to(device)
    model.eval()
    return model


def create_fake_inputs(batch_size: int, seq_length: int, hidden_size: int) -> dict:
    """
    Create a fake input dictionary matching what the inversion model expects.
    We include:
      - hypothesis_embedding: (B, hidden_size)
      - frozen_embeddings: (B, hidden_size)
      - hypothesis_input_ids: (B, seq_length)
      - hypothesis_attention_mask: (B, seq_length)
      - embedder_input_ids: (B, seq_length)
      - embedder_attention_mask: (B, seq_length)
      - input_ids: (B, seq_length)         [decoder inputs]
      - attention_mask: (B, seq_length)      [decoder attention mask]
      - labels: (B, seq_length)
    """
    hypothesis_embedding = torch.randn(batch_size, hidden_size, device=device)
    frozen_embeddings = torch.randn(batch_size, hidden_size, device=device)
    # Assume vocabulary size ~32128 (adjust if needed)
    hypothesis_input_ids = torch.randint(0, 32128, (batch_size, seq_length), device=device)
    hypothesis_attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long, device=device)
    embedder_input_ids = torch.randint(0, 32128, (batch_size, seq_length), device=device)
    embedder_attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long, device=device)
    # For simplicity, set decoder inputs equal to embedder inputs.
    input_ids = embedder_input_ids.clone()
    attention_mask = embedder_attention_mask.clone()
    labels = hypothesis_input_ids.clone()
    return {
        "hypothesis_embedding": hypothesis_embedding,
        "frozen_embeddings": frozen_embeddings,
        "hypothesis_input_ids": hypothesis_input_ids,
        "hypothesis_attention_mask": hypothesis_attention_mask,
        "embedder_input_ids": embedder_input_ids,
        "embedder_attention_mask": embedder_attention_mask,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


@pytest.fixture
def fake_inputs(model_with_vq) -> dict:
    # IMPORTANT: use the embedder's hidden size (e.g. 768 for bert-base-uncased)
    hidden_size = model_with_vq.embedder.config.hidden_size
    return create_fake_inputs(batch_size=2, seq_length=8, hidden_size=hidden_size)


def test_forward_loss_shapes(model_with_vq, fake_inputs):
    """
    Test that a forward pass through the inversion model (with VQ enabled)
    produces a scalar loss and outputs with the expected batch dimensions.
    """
    output = model_with_vq(**fake_inputs)
    loss = output.loss
    # Check that loss is a scalar.
    assert loss.dim() == 0, "Loss should be a scalar"
    assert torch.isfinite(loss), "Loss should be finite"
    # If logits are returned, verify the batch dimension.
    if hasattr(output, "logits"):
        batch_size = fake_inputs["hypothesis_input_ids"].shape[0]
        assert output.logits.shape[0] == batch_size, "Mismatch in logits batch dimension"


def test_vq_loss_effect(model_with_vq, model_without_vq, fake_inputs):
    """
    Compare the loss of the inversion model when VQ is enabled versus disabled.
    We expect a difference since the VQ branch introduces an extra loss term.
    """
    output_vq = model_with_vq(**fake_inputs)
    output_no_vq = model_without_vq(**fake_inputs)
    loss_vq = output_vq.loss.item()
    loss_no_vq = output_no_vq.loss.item()
    assert abs(loss_vq - loss_no_vq) > 1e-4, "Loss with VQ should differ from loss without VQ"


def test_backward_pass_gradients(model_with_vq, fake_inputs):
    """
    Verify that gradients flow back through the inversion model with VQ enabled,
    and specifically that the codebook parameters receive nonzero gradients.
    """
    inputs = {
        k: v.clone().detach().requires_grad_(True) if v.dtype == torch.float32 else v
        for k, v in fake_inputs.items()
    }
    output = model_with_vq(**inputs)
    loss = output.loss
    loss.backward()

    # Check that at least one parameter in the embedding transformation receives a gradient.
    grad_found = False
    for name, param in model_with_vq.embedding_transform.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for embedding_transform parameter {name}"
            grad_found = True
    assert grad_found, "No gradient found in embedding_transform parameters"

    # If VQ is enabled, ensure the codebook receives nonzero gradients.
    if model_with_vq.use_vq:
        codebook_grad = model_with_vq.vector_quantizer.codebook.grad
        assert codebook_grad is not None, "No gradient computed for the codebook"
        assert torch.any(codebook_grad != 0), "Codebook gradients are all zero"
