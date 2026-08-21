"""
autoencoder.py — Semantic Embedding Autoencoder
================================================
Architecture:
  Encoder: 768 → 512 → 256 → 256 (latent)
  Decoder: 256 → 256 → 512 → 768

Purpose (Advanced ML Component):
  - Compresses high-dimensional Sentence-BERT embeddings (BGE-base: 768-dim)
  - Forces the network to learn a compact semantic representation
  - Denoises embeddings by discarding low-variance dimensions
  - The 256-dim latent vectors are fed to the BART summarizer
    to provide semantically rich, noise-reduced sentence selection

Training Objective:
  Minimize Mean Squared Error between input embeddings and
  reconstructed embeddings (standard autoencoder loss).

Evaluation Metrics:
  - Reconstruction loss (MSE)
  - Cosine similarity between original and reconstructed embeddings
  - ROUGE score improvement over baseline summarization (without AE)

Usage:
  ae = SemanticAutoencoder(input_dim=768, latent_dim=256)
  ae.train_on_corpus(sentences)          # Fine-tune on your data
  compressed = ae.encode(embeddings)     # (n, 768) → (n, 256)

Course: Advanced Topics in Machine Learning (HTML)
"""

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

WEIGHTS_PATH = Path(__file__).parent.parent / "models" / "autoencoder_weights.pt"


class SemanticAutoencoder:
    """
    PyTorch autoencoder for semantic embedding compression.
    Falls back to PCA if PyTorch is unavailable.
    """

    def __init__(self, input_dim: int = 384, latent_dim: int = 128):
        self.input_dim  = input_dim
        self.latent_dim = latent_dim
        self.model      = None
        self.pca_fallback = None
        self._build_model()

    # ─── ARCHITECTURE ─────────────────────────────────────────────
    def _build_model(self):
        try:
            import torch
            import torch.nn as nn

            class _AE(nn.Module):
                def __init__(self, in_dim, lat_dim):
                    super().__init__()
                    # Encoder: progressively compress
                    self.encoder = nn.Sequential(
                        nn.Linear(in_dim, 512),
                        nn.ReLU(),
                        nn.BatchNorm1d(512),
                        nn.Dropout(0.1),

                        nn.Linear(512, 256),
                        nn.ReLU(),
                        nn.BatchNorm1d(256),
                        nn.Dropout(0.1),

                        nn.Linear(256, lat_dim),
                        nn.Tanh()          # Bounded latent space [-1, 1]
                    )
                    # Decoder: reconstruct original dimension
                    self.decoder = nn.Sequential(
                        nn.Linear(lat_dim, 256),
                        nn.ReLU(),
                        nn.BatchNorm1d(256),

                        nn.Linear(256, 512),
                        nn.ReLU(),
                        nn.BatchNorm1d(512),

                        nn.Linear(512, in_dim)
                        # No activation — embeddings are unbounded floats
                    )

                def forward(self, x):
                    latent = self.encoder(x)
                    recon  = self.decoder(latent)
                    return recon, latent

                def encode(self, x):
                    return self.encoder(x)

            self.model = _AE(self.input_dim, self.latent_dim)
            log.info(f"Autoencoder built: {self.input_dim} → {self.latent_dim} → {self.input_dim}")

        except ImportError:
            log.warning("PyTorch not available. PCA fallback will be used.")

    # ─── WEIGHTS ──────────────────────────────────────────────────
    def try_load_weights(self, validate: bool = True):
        """Load pre-trained weights if available.
        
        Args:
            validate: If True, run a quick validation to ensure weights are trained (not random).
        """
        if self.model is None: return
        if WEIGHTS_PATH.exists():
            try:
                import torch
                self.model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
                self.model.eval()
                log.info(f"Loaded autoencoder weights from {WEIGHTS_PATH}")

                if validate:
                    self._validate_weights()
            except Exception as e:
                log.warning(f"Could not load weights: {e}. Using random init.")
                self._init_weights()
        else:
            log.info("No pre-trained weights found. Using random initialization.")
            if self.model: self.model.eval()

    def _validate_weights(self):
        """Quick validation that weights produce reasonable reconstructions."""
        if self.model is None:
            return
        try:
            import torch
            import torch.nn as nn

            # Create synthetic normalized embeddings (like BGE output)
            with torch.no_grad():
                test_emb = torch.randn(50, self.input_dim)
                test_emb = nn.functional.normalize(test_emb, p=2, dim=1)

                self.model.eval()
                recon, _ = self.model(test_emb)
                mse = nn.functional.mse_loss(recon, test_emb).item()
                cos_sim = nn.functional.cosine_similarity(recon, test_emb, dim=1).mean().item()

                # Random init typically gives MSE ~1.0 and cos_sim ~0.0
                # Trained weights should give MSE < 0.1 and cos_sim > 0.5
                if mse > 0.5 or cos_sim < 0.3:
                    log.warning(f"Loaded weights may be undertrained (MSE={mse:.4f}, cos_sim={cos_sim:.4f}). "
                                f"Consider retraining with: python -m backend.autoencoder --train-corpus <file>")
                else:
                    log.info(f"Weight validation passed: MSE={mse:.6f}, cos_sim={cos_sim:.4f}")
        except Exception as e:
            log.debug(f"Weight validation skipped: {e}")

    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization."""
        if self.model is None:
            return
        try:
            import torch.nn as nn
            def init_layer(m):
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            self.model.apply(init_layer)
            log.info("Autoencoder weights initialized with Xavier initialization")
        except Exception as e:
            log.warning(f"Weight initialization failed: {e}")

    def save_weights(self):
        if self.model is None: return
        import torch
        WEIGHTS_PATH.parent.mkdir(exist_ok=True)
        torch.save(self.model.state_dict(), WEIGHTS_PATH)
        log.info(f"Saved autoencoder weights → {WEIGHTS_PATH}")

    # ─── TRAINING ─────────────────────────────────────────────────
    def train_on_embeddings(self,
                             embeddings: np.ndarray,
                             epochs: int = 50,
                             batch_size: int = 32,
                             lr: float = 1e-3) -> list:
        """
        Train the autoencoder on a batch of sentence embeddings.

        Args:
            embeddings: np.ndarray of shape (N, input_dim)
            epochs:     number of training epochs
            batch_size: mini-batch size
            lr:         learning rate

        Returns:
            list of per-epoch training losses

        Training procedure:
          1. Shuffle embeddings each epoch
          2. Forward pass → encoder latent → decoder reconstruction
          3. MSE loss between input and reconstruction
          4. Backprop + Adam optimizer step
        """
        if self.model is None:
            log.warning("No PyTorch model — skipping training.")
            return []

        import torch
        import torch.nn as nn
        import torch.optim as optim

        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

        X = torch.FloatTensor(embeddings)
        losses = []

        for epoch in range(epochs):
            # Shuffle
            perm = torch.randperm(len(X))
            X = X[perm]
            epoch_loss = 0.0
            n_batches  = 0

            for i in range(0, len(X), batch_size):
                batch = X[i: i + batch_size]
                optimizer.zero_grad()
                recon, _ = self.model(batch)
                loss = criterion(recon, batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches  += 1

            scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)
            losses.append(avg_loss)

            if epoch % 10 == 0 or epoch == epochs - 1:
                log.info(f"AE Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.6f}")

        self.model.eval()
        self.save_weights()
        return losses

    def train_on_corpus(self, sentences: list, **kwargs):
        """Convenience method: encode sentences → train AE."""
        from sentence_transformers import SentenceTransformer
        sbert = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = sbert.encode(sentences, show_progress_bar=False, convert_to_numpy=True)
        return self.train_on_embeddings(embeddings, **kwargs)

    # ─── INFERENCE ────────────────────────────────────────────────
    def encode(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compress embeddings: (N, input_dim) → (N, latent_dim).
        Falls back to PCA if PyTorch is unavailable.
        """
        if self.model is not None:
            try:
                import torch
                with torch.no_grad():
                    x = torch.FloatTensor(embeddings)
                    latent = self.model.encode(x)
                    return latent.numpy()
            except Exception as e:
                log.warning(f"AE encode error: {e}. Using PCA fallback.")

        return self._pca_encode(embeddings)

    def decode(self, latent: np.ndarray) -> np.ndarray:
        """Reconstruct embeddings from latent space."""
        if self.model is not None:
            try:
                import torch
                with torch.no_grad():
                    z = torch.FloatTensor(latent)
                    recon = self.model.decoder(z)
                    return recon.numpy()
            except Exception as e:
                log.warning(f"AE decode error: {e}")
        return latent  # Cannot reconstruct without model

    def _pca_encode(self, embeddings: np.ndarray) -> np.ndarray:
        """PCA-based dimensionality reduction as a fallback."""
        try:
            from sklearn.decomposition import PCA
            if self.pca_fallback is None:
                self.pca_fallback = PCA(n_components=min(self.latent_dim,
                                                          embeddings.shape[1]))
                self.pca_fallback.fit(embeddings)
            return self.pca_fallback.transform(embeddings)
        except Exception:
            # Last resort: truncate
            return embeddings[:, :self.latent_dim]

    # ─── EVALUATION ───────────────────────────────────────────────
    def evaluate(self, embeddings: np.ndarray) -> dict:
        """
        Compute evaluation metrics:
          - reconstruction_mse: how well decoder reconstructs input
          - cosine_similarity:  semantic preservation metric
          - compression_ratio:  input_dim / latent_dim
        """
        compressed  = self.encode(embeddings)
        reconstructed = self.decode(compressed)

        mse = float(np.mean((embeddings - reconstructed) ** 2))

        from sklearn.metrics.pairwise import cosine_similarity as cos_sim
        sims = [
            cos_sim(embeddings[i:i+1], reconstructed[i:i+1])[0][0]
            for i in range(min(len(embeddings), 100))
        ]
        avg_cosine = float(np.mean(sims))

        return {
            "reconstruction_mse":    round(mse, 6),
            "avg_cosine_similarity": round(avg_cosine, 4),
            "compression_ratio":     round(self.input_dim / self.latent_dim, 2),
            "input_dim":             self.input_dim,
            "latent_dim":            self.latent_dim
        }

    def __repr__(self):
        return (f"SemanticAutoencoder("
                f"input={self.input_dim}, latent={self.latent_dim}, "
                f"pytorch={'available' if self.model else 'unavailable'})")


# ─── TRAINING CLI ───────────────────────────────────────────────
def _train_from_corpus_file(corpus_path: str, epochs: int = 50, lr: float = 1e-3, batch_size: int = 64):
    """Train autoencoder on sentences extracted from a text file."""
    import logging
    logging.basicConfig(level=logging.INFO)

    from pathlib import Path
    path = Path(corpus_path)
    if not path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    # Extract sentences from file
    import nltk
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    text = path.read_text(encoding="utf-8")
    sentences = nltk.sent_tokenize(text)
    # Filter: keep sentences with 10-100 words
    sentences = [s.strip() for s in sentences if 10 <= len(s.split()) <= 100]

    log.info(f"Loaded {len(sentences)} sentences from {corpus_path}")
    if len(sentences) < 100:
        log.warning(f"Small corpus ({len(sentences)} sentences). Consider adding more data.")

    ae = SemanticAutoencoder(input_dim=384, latent_dim=128)
    print(f"Architecture: {ae}")
    losses = ae.train_on_corpus(sentences, epochs=epochs, lr=lr, batch_size=batch_size)
    print(f"\nFinal training loss: {losses[-1]:.6f}")

    # Evaluate
    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    test_emb = sbert.encode(sentences[:min(100, len(sentences))], convert_to_numpy=True)
    metrics = ae.evaluate(test_emb)
    print("\nEvaluation Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    return ae


def _train_from_documents(doc_dir: str, epochs: int = 50, lr: float = 1e-3, batch_size: int = 64):
    """Train autoencoder on all PDF/PPTX files in a directory."""
    import logging
    logging.basicConfig(level=logging.INFO)

    from pathlib import Path

    import nltk

    from backend.utils import extract_text_pdf, extract_text_pptx

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    path = Path(doc_dir)
    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {doc_dir}")

    all_sentences = []
    for file_path in path.rglob("*"):
        if file_path.suffix.lower() in {".pdf", ".pptx", ".ppt"}:
            try:
                raw = file_path.read_bytes()
                if file_path.suffix.lower() == ".pdf":
                    text = extract_text_pdf(raw)
                else:
                    text = extract_text_pptx(raw)

                sentences = nltk.sent_tokenize(text)
                sentences = [s.strip() for s in sentences if 10 <= len(s.split()) <= 100]
                all_sentences.extend(sentences)
                log.info(f"Extracted {len(sentences)} sentences from {file_path.name}")
            except Exception as e:
                log.warning(f"Failed to process {file_path}: {e}")

    log.info(f"Total sentences from documents: {len(all_sentences)}")

    ae = SemanticAutoencoder(input_dim=384, latent_dim=128)
    print(f"Architecture: {ae}")
    losses = ae.train_on_corpus(all_sentences, epochs=epochs, lr=lr, batch_size=batch_size)
    print(f"\nFinal training loss: {losses[-1]:.6f}")

    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    test_emb = sbert.encode(all_sentences[:min(100, len(all_sentences))], convert_to_numpy=True)
    metrics = ae.evaluate(test_emb)
    print("\nEvaluation Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    return ae


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Train SemanticAutoencoder")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train from text file
    file_parser = subparsers.add_parser("train-corpus", help="Train on a text corpus file")
    file_parser.add_argument("corpus_path", help="Path to text file (one sentence per line or full text)")
    file_parser.add_argument("--epochs", type=int, default=50)
    file_parser.add_argument("--lr", type=float, default=1e-3)
    file_parser.add_argument("--batch-size", type=int, default=64)

    # Train from documents directory
    doc_parser = subparsers.add_parser("train-docs", help="Train on PDF/PPTX files in a directory")
    doc_parser.add_argument("doc_dir", help="Path to directory with documents")
    doc_parser.add_argument("--epochs", type=int, default=50)
    doc_parser.add_argument("--lr", type=float, default=1e-3)
    doc_parser.add_argument("--batch-size", type=int, default=64)

    # Quick test (synthetic)
    test_parser = subparsers.add_parser("quick-test", help="Quick synthetic training test")
    test_parser.add_argument("--epochs", type=int, default=10)

    args = parser.parse_args()

    if args.command == "train-corpus":
        _train_from_corpus_file(args.corpus_path, args.epochs, args.lr, args.batch_size)
    elif args.command == "train-docs":
        _train_from_documents(args.doc_dir, args.epochs, args.lr, args.batch_size)
    elif args.command == "quick-test":
        import logging
        logging.basicConfig(level=logging.INFO)

        sample_sentences = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses multi-layer neural networks.",
            "Supervised learning requires labeled training data.",
            "Unsupervised learning finds patterns without labels.",
            "Transformers use self-attention mechanisms.",
            "BERT is a bidirectional encoder representation.",
            "GPT models are autoregressive language models.",
            "Gradient descent optimizes model parameters.",
            "Overfitting occurs when models memorize training data.",
            "Regularization techniques prevent overfitting.",
            "The vanishing gradient problem affects deep networks.",
            "Convolutional networks excel at image recognition.",
            "Recurrent networks handle sequential data.",
            "Attention mechanisms improve sequence-to-sequence models.",
            "Transfer learning leverages pre-trained representations.",
            "Fine-tuning adapts pre-trained models to new tasks.",
            "Autoencoders learn compressed data representations.",
            "Variational autoencoders generate new data samples.",
            "GANs use adversarial training for generation.",
            "Reinforcement learning optimizes cumulative reward.",
        ] * 20

        ae = SemanticAutoencoder(input_dim=384, latent_dim=128)
        print(f"Architecture: {ae}")
        losses = ae.train_on_corpus(sample_sentences, epochs=args.epochs, lr=1e-3)
        print(f"\nFinal training loss: {losses[-1]:.6f}")

        from sentence_transformers import SentenceTransformer
        sbert = SentenceTransformer("all-MiniLM-L6-v2")
        test_emb = sbert.encode(sample_sentences[:20], convert_to_numpy=True)
        metrics = ae.evaluate(test_emb)
        print("\nEvaluation Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
    else:
        parser.print_help()
        sys.exit(1)
