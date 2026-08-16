"""
init_autoencoder_weights.py — Initialize Autoencoder Weights Without External Dependencies
=========================================================================================
Creates and saves initial weights for the SemanticAutoencoder using only PyTorch.
This avoids the sentence_transformers/transformers version conflicts.
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.autoencoder import SemanticAutoencoder


def init_weights():
    """Initialize and save autoencoder weights."""
    print("Initializing SemanticAutoencoder (384 -> 128)...")
    
    ae = SemanticAutoencoder(input_dim=384, latent_dim=128)
    
    if ae.model is None:
        print("ERROR: PyTorch model not available")
        return False
    
    # Initialize with Xavier/Glorot initialization for better starting point
    def init_layer(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    ae.model.apply(init_layer)
    
    # Train on random data for a few epochs to get reasonable weights
    print("Training on synthetic data for initialization...")
    ae.model.train()
    
    # Generate synthetic embeddings that mimic BGE distribution
    # BGE embeddings are normalized, so they lie on a hypersphere
    torch.manual_seed(42)
    n_samples = 2000
    # Sample from normal and normalize to unit sphere
    synthetic_embeddings = torch.randn(n_samples, 384)
    synthetic_embeddings = nn.functional.normalize(synthetic_embeddings, p=2, dim=1)
    
    optimizer = torch.optim.Adam(ae.model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    batch_size = 64
    n_epochs = 20
    
    for epoch in range(n_epochs):
        # Shuffle
        perm = torch.randperm(n_samples)
        synthetic_embeddings = synthetic_embeddings[perm]
        
        epoch_loss = 0.0
        n_batches = 0
        
        for i in range(0, n_samples, batch_size):
            batch = synthetic_embeddings[i:i+batch_size]
            optimizer.zero_grad()
            recon, _ = ae.model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ae.model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        
        if epoch % 5 == 0 or epoch == n_epochs - 1:
            print(f"  Epoch {epoch+1:3d}/{n_epochs} | Loss: {avg_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.2e}")
    
    ae.model.eval()
    
    # Save weights
    weights_path = Path(__file__).parent.parent / "models" / "autoencoder_weights.pt"
    weights_path.parent.mkdir(exist_ok=True)
    torch.save(ae.model.state_dict(), weights_path)
    
    print(f"\nWeights saved to: {weights_path}")
    print(f"File size: {weights_path.stat().st_size / 1024:.1f} KB")
    
    # Quick evaluation
    print("\nEvaluating...")
    with torch.no_grad():
        test_emb = synthetic_embeddings[:100]
        recon, latent = ae.model(test_emb)
        mse = nn.functional.mse_loss(recon, test_emb).item()
        cos_sim = nn.functional.cosine_similarity(recon, test_emb, dim=1).mean().item()
        print(f"  Reconstruction MSE: {mse:.6f}")
        print(f"  Cosine Similarity: {cos_sim:.4f}")
        print(f"  Compression Ratio: {384/128:.2f}x")
        print(f"  Latent shape: {latent.shape}")
    
    return True


if __name__ == "__main__":
    success = init_weights()
    if success:
        print("\n✅ Autoencoder weights initialized successfully!")
    else:
        print("\n❌ Failed to initialize weights")
        sys.exit(1)