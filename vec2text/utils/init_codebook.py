from sklearn.cluster import KMeans
import torch

torch.manual_seed(0)


def initialize_codebook(embeddings: torch.Tensor, num_embeddings: int) -> torch.Tensor:
    """
    Runs k-means on the provided embeddings to obtain centroids.

    Args:
        embeddings: A tensor of shape (N, D) where N is the number of samples
                    and D is the embedding dimensionality.
        num_embeddings: The desired number of codebook vectors (clusters).

    Returns:
        A tensor of shape (num_embeddings, D) containing the computed centroids.
    """
    kmeans = KMeans(n_clusters=num_embeddings, random_state=0).fit(embeddings.cpu().numpy())
    centers = torch.tensor(
        kmeans.cluster_centers_, dtype=embeddings.dtype, device=embeddings.device
    )
    return centers


def codebook_from_dataset(train_dataset, num_samples: int = 2**13, codebook_size: int = 412):
    # Determine how many samples to take (either all or up to num_samples)
    sample_size = min(len(train_dataset), num_samples)

    # Randomly sample indices from the dataset
    indices = torch.randperm(len(train_dataset))[:sample_size].tolist()

    # Get the samples
    all_embeddings = train_dataset["frozen_embeddings"][indices]

    # Initialize codebook with k-means
    return initialize_codebook(all_embeddings, codebook_size)


def initialize_model_codebook_from_dataset(
    model, train_dataset, num_samples: int = 2**11, debug: bool = False
) -> None:
    """
    Samples a subset of training embeddings from the provided dataset,
    runs k-means clustering, and reinitializes the model's codebook with the centroids.

    Args:
        model: The inversion model instance (which must have vector_quantizer).
        train_dataset: The training Dataset.
        num_samples: The maximum number of embeddings to sample for clustering.
    """
    codebook_size = model.vector_quantizer.num_embeddings
    centers = codebook_from_dataset(
        train_dataset=train_dataset,
        num_samples=num_samples,
        codebook_size=codebook_size,
    )
    assert centers.shape[0] == codebook_size, "not enough centers!"
    # Debug the distances!
    if debug:
        from torchmetrics.functional.pairwise import pairwise_cosine_similarity

        distances = pairwise_cosine_similarity(train_dataset["frozen_embeddings"], centers)
        unique_minimums = torch.argmin(distances, dim=1).unique().shape
        assert unique_minimums[0] > 100, f"{unique_minimums=} has too few matches!"
        # assert unique_minimums[0] == codebook_size, f"Sizes don't match!: {unique_minimums} versus {codebook_size=}"

    model.vector_quantizer.codebook.data.copy_(centers)
    # assert torch.all(centers.to("cpu") == model.vector_quantizer.codebook.data.to("cpu")), "Mismatch in codebook!"
    model.training_args = getattr(model, "training_args", None)

    print(f"Initialized codebook with k-means centroids from {num_samples} samples.")
