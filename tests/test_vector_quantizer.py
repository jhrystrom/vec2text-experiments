import torch

from vec2text.models.vector_quantizer import VectorQuantizer


def test_vector_quantizer_output_shape() -> None:
    batch_size = 8
    embedding_dim = 64
    num_embeddings = 512
    vq = VectorQuantizer(num_embeddings, embedding_dim)
    inputs = torch.randn(batch_size, embedding_dim)
    quantized, loss = vq(inputs)
    assert quantized.shape == (batch_size, embedding_dim)
    assert loss.item() >= 0
