#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D
import transformers

from vec2text.models import InversionModel
from vec2text.utils.init_codebook import initialize_model_codebook_from_dataset
from vec2text.utils import dataset_map_multi_worker
import vec2text.analyze_utils
from vec2text.data_helpers import dataset_from_args
from vec2text.run_args import DataArguments


def find_most_recent_model():
    """Find the most recent model checkpoint in the saves directory"""
    # Look in the saves directory for all model checkpoints
    save_dirs = glob.glob('saves/*/')
    
    if not save_dirs:
        raise FileNotFoundError("No model checkpoints found in the 'saves' directory")
    
    # Sort by modification time (most recent last)
    latest_dir = max(save_dirs, key=os.path.getmtime)
    print(f"Found most recent model checkpoint: {latest_dir}")
    
    # Check for a specific checkpoint
    checkpoint_dirs = glob.glob(os.path.join(latest_dir, 'checkpoint-*'))
    if checkpoint_dirs:
        # Use the latest checkpoint
        latest_checkpoint = max(checkpoint_dirs, key=os.path.getmtime)
        return latest_checkpoint
    
    # If no specific checkpoint, return the main directory
    return latest_dir


def save_codebook_state(model):
    """Extract and save the current state of the codebook"""
    if not hasattr(model, 'vector_quantizer') or model.vector_quantizer is None:
        raise ValueError("Model does not have a vector quantizer")
    
    return model.vector_quantizer.codebook.detach().cpu()


def main():
    # Create output directory
    output_dir = 'codebook_comparison'
    os.makedirs(output_dir, exist_ok=True)
    
    # Find the most recent model
    model_path = find_most_recent_model()
    print(f"Loading model from {model_path}")
    
    # Load the model
    model, args_dict = vec2text.analyze_utils.load_model_from_pretrained(model_path)
    model.eval()
    
    # Save the current (trained) codebook state
    print("Extracting trained codebook state")
    trained_codebook = save_codebook_state(model)
    
    # Configure dataset for k-means initialization
    # Default num_samples for k-means initialization
    num_samples = 2048
    
    # Get dataset name from the loaded model if available
    data_args = args_dict.get("data_args", None)
    if data_args is None:
        dataset_name = 'c4'  # Default dataset
        data_args = DataArguments(dataset_name=dataset_name)
    
    print(f"Using dataset {data_args.dataset_name} for k-means initialization")
    
    # Load dataset for k-means initialization
    dataset = dataset_from_args(data_args)
    train_dataset = dataset["train"]
    
    # Create a fresh copy of the model with reinitialized codebook
    print("Creating a fresh model with reinitialized codebook")
    config = model.config
    fresh_model = InversionModel(config)
    
    # Initialize k-means codebook
    print(f"Initializing k-means codebook with {num_samples} samples")
    initialize_model_codebook_from_dataset(fresh_model, train_dataset, num_samples=num_samples)
    
    # Save the initial codebook state
    initial_codebook = save_codebook_state(fresh_model)
    
    # Apply PCA to reduce dimensionality for visualization
    # Use 2 components for 2D plots only
    n_components = 2
    print(f"Applying PCA to reduce dimensionality to {n_components} components")
    pca = PCA(n_components=n_components)
    
    # Concatenate both codebooks and fit PCA
    combined_codebook = torch.cat([initial_codebook, trained_codebook], dim=0)
    pca.fit(combined_codebook.numpy())
    
    # Transform codebooks to lower-dimensional space
    initial_codebook_pca = pca.transform(initial_codebook.numpy())
    trained_codebook_pca = pca.transform(trained_codebook.numpy())
    
    # Compute statistics about the movement of codebook vectors
    distances = np.sqrt(np.sum((initial_codebook_pca - trained_codebook_pca) ** 2, axis=1))
    mean_distance = np.mean(distances)
    median_distance = np.median(distances)
    max_distance = np.max(distances)
    min_distance = np.min(distances)
    
    print(f"Movement statistics in PCA space:")
    print(f"  Mean distance: {mean_distance:.4f}")
    print(f"  Median distance: {median_distance:.4f}")
    print(f"  Max distance: {max_distance:.4f}")
    print(f"  Min distance: {min_distance:.4f}")
    
    # Create main comparison plot (2D)
    plt.figure(figsize=(12, 10))
    
    # Plot initial codebook points
    plt.scatter(initial_codebook_pca[:, 0], initial_codebook_pca[:, 1], 
                c='blue', alpha=0.7, label='Initial K-means Centers')
    
    # Plot trained codebook points
    plt.scatter(trained_codebook_pca[:, 0], trained_codebook_pca[:, 1], 
                c='red', alpha=0.7, label='Updated Centers After Training')
    
    # Draw lines connecting corresponding points
    for i in range(len(initial_codebook_pca)):
        plt.plot([initial_codebook_pca[i, 0], trained_codebook_pca[i, 0]],
                 [initial_codebook_pca[i, 1], trained_codebook_pca[i, 1]],
                 'k-', alpha=0.2)
    
    plt.title('Comparison of Initial K-means Centers vs. Updated Centers After Training')
    plt.xlabel(f'PCA Component 1 (Explained Variance: {pca.explained_variance_ratio_[0]:.4f})')
    plt.ylabel(f'PCA Component 2 (Explained Variance: {pca.explained_variance_ratio_[1]:.4f})')
    
    # Add custom legend with distance statistics
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Initial K-means Centers'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Updated Centers'),
        Line2D([0], [0], color='k', alpha=0.2, label='Movement'),
    ]
    
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Add text box with distance statistics
    stats_text = (f"Distance Statistics:\n"
                  f"Mean: {mean_distance:.4f}\n"
                  f"Median: {median_distance:.4f}\n"
                  f"Max: {max_distance:.4f}\n"
                  f"Min: {min_distance:.4f}")
    
    plt.figtext(0.02, 0.02, stats_text, fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    # Save the plot
    output_path = os.path.join(output_dir, "codebook_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to {output_path}")
    
    # Create a histogram of distances
    plt.figure(figsize=(10, 6))
    plt.hist(distances, bins=30, alpha=0.7, color='purple')
    plt.title('Histogram of Distances Between Initial and Updated Codebook Vectors')
    plt.xlabel('Distance in PCA Space')
    plt.ylabel('Frequency')
    
    # Add vertical lines for statistics
    plt.axvline(mean_distance, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_distance:.4f}')
    plt.axvline(median_distance, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_distance:.4f}')
    
    plt.legend()
    
    # Save the histogram
    hist_path = os.path.join(output_dir, "distance_histogram.png")
    plt.savefig(hist_path, dpi=300, bbox_inches='tight')
    print(f"Saved distance histogram to {hist_path}")
    
    
    # Save the PCA objects and codebook data for further analysis
    save_data = {
        'model_path': model_path,
        'initial_codebook': initial_codebook.numpy(),
        'trained_codebook': trained_codebook.numpy(),
        'initial_codebook_pca': initial_codebook_pca,
        'trained_codebook_pca': trained_codebook_pca,
        'pca': pca,
        'distances': distances,
        'stats': {
            'mean_distance': mean_distance,
            'median_distance': median_distance,
            'max_distance': max_distance,
            'min_distance': min_distance,
        }
    }
    
    torch.save(save_data, os.path.join(output_dir, "codebook_comparison_data.pt"))
    print(f"Saved comparison data to {os.path.join(output_dir, 'codebook_comparison_data.pt')}")
    
    print("Codebook comparison complete!")


if __name__ == "__main__":
    main()