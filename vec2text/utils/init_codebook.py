import os
import wandb
import logging
import numpy as np

from sklearn.cluster import KMeans
import torch

logger = logging.getLogger(__name__)
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
    centers = torch.tensor(kmeans.cluster_centers_, dtype=embeddings.dtype, device=embeddings.device)
    return centers


def initialize_model_codebook_from_dataset(model, train_dataset, num_samples: int = 1024) -> None:
    """
    Samples a subset of training embeddings from the provided dataset,
    runs k-means clustering, and reinitializes the model's codebook with the centroids.
    
    Args:
        model: The inversion model instance (which must have vector_quantizer).
        train_dataset: The training Dataset.
        num_samples: The maximum number of embeddings to sample for clustering.
    """
    # Determine how many samples to take (either all or up to num_samples)
    sample_size = min(len(train_dataset), num_samples)
    
    # Randomly sample indices from the dataset
    indices = torch.randperm(len(train_dataset))[:sample_size].tolist()
    
    # Get the samples
    embeddings_list = []
    device = next(model.parameters()).device
    
    for idx in indices:
        sample = train_dataset[idx]
        
        if "frozen_embeddings" not in sample:
            raise ValueError("This function requires 'frozen_embeddings' to be present in the dataset. "
                           "Make sure you're using a dataset with precomputed embeddings.")
        
        # Get the frozen embedding and move to device
        embedding = sample["frozen_embeddings"].to(device).unsqueeze(0)
        embeddings_list.append(embedding)
    
    # Concatenate all embeddings
    all_embeddings = torch.cat(embeddings_list, dim=0)
    
    # Initialize codebook with k-means
    centers = initialize_codebook(all_embeddings, model.vector_quantizer.num_embeddings)
    model.vector_quantizer.codebook.data.copy_(centers)

    model.training_args = getattr(model, 'training_args', None)
    log_codebook(
        model,
        path=os.path.join(getattr(model, 'training_args', None).output_dir 
                            if getattr(model, 'training_args', None) is not None else ".", 
                            "initial_codebook.npy"),
        prefix="initial_"
    )

    
    print(f"Initialized codebook with k-means centroids from {all_embeddings.shape[0]} samples.")


def log_codebook(model, path=None, prefix=""):
    """Log stats about the VQ codebook and optionally save it to disk"""
    if not hasattr(model, "use_vq") or not model.use_vq or not hasattr(model, "vector_quantizer"):
        return
    
    codebook = model.vector_quantizer.codebook.detach().cpu().numpy()
    
    # Log basic stats
    logger.info(f"{prefix}Codebook shape: {codebook.shape}, mean: {np.mean(codebook):.4f}, "
                f"std: {np.std(codebook):.4f}, min: {np.min(codebook):.4f}, max: {np.max(codebook):.4f}")
    
    # Save to disk if path is provided
    if path is not None:
        np.save(path, codebook)
        logger.info(f"Saved {prefix}codebook to {path}")

    # Log to wandb if enabled
    if hasattr(model, 'training_args') and model.training_args.use_wandb:
        wandb.log({
            f"{prefix}codebook/shape": list(codebook.shape),
            f"{prefix}codebook/histogram": wandb.Histogram(codebook.reshape(-1))
        })