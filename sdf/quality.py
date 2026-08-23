"""Quality measurement for synthetic images: fidelity, memorization, degeneracy.

Three questions, in order of importance:

1. **Memorization** - did the generator reproduce a real training image? A
   synthetic dataset that copies real patients is worse than useless. Measured
   as max cosine similarity between each synthetic embedding and every train
   embedding of the same class; anything above the threshold is a defect.
2. **Degeneracy** - blank/black/collapsed outputs, measured cheaply from pixel
   statistics before any model is loaded.
3. **Fidelity** - FID between synthetic and real held-out images of the class.
   Diagnostic, not a gate on its own at smoke scale (FID needs hundreds of
   samples to be stable; we report it with the sample count attached).

Embeddings come from torchvision's ImageNet Inception v3 (present in the
Kaggle image). FID is the standard Frechet distance between Gaussian fits;
the matrix sqrt uses eigendecomposition to avoid a scipy dependency.

Heavy imports stay inside functions - the control machine has no torch.
"""

from __future__ import annotations

from pathlib import Path

MEMORIZATION_THRESHOLD = 0.985
MIN_STD = 4.0          # 0-255 scale; below this the image is essentially flat
FID_UNSTABLE_BELOW = 200


# ------------------------------------------------------------------ degeneracy
def pixel_screen(paths: list[Path]) -> dict:
    """Cheap PIL-only screen: flat/blank images and exact-duplicate hashes."""
    import hashlib

    from PIL import Image, ImageStat

    flat: list[str] = []
    hashes: dict[str, str] = {}
    duplicates: list[tuple[str, str]] = []
    for p in paths:
        with Image.open(p) as img:
            img = img.convert("RGB")
            stat = ImageStat.Stat(img)
            if max(stat.stddev) < MIN_STD:
                flat.append(p.name)
            digest = hashlib.sha256(img.tobytes()).hexdigest()
        if digest in hashes:
            duplicates.append((hashes[digest], p.name))
        else:
            hashes[digest] = p.name
    return {
        "checked": len(paths),
        "flat": flat,
        "exact_duplicates": duplicates,
    }


# ------------------------------------------------------------------ embeddings
def embed_images(paths: list[Path], batch_size: int = 16):
    """Return an (N, 2048) float32 numpy array of Inception-v3 pool features."""
    import numpy as np
    import torch
    from PIL import Image
    from torchvision.models import Inception_V3_Weights, inception_v3

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = Inception_V3_Weights.IMAGENET1K_V1
    model = inception_v3(weights=weights)
    model.fc = torch.nn.Identity()  # expose the 2048-d pool features
    model.eval().to(device)
    preprocess = weights.transforms()

    feats = []
    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            batch = []
            for p in paths[start : start + batch_size]:
                with Image.open(p) as img:
                    batch.append(preprocess(img.convert("RGB")))
            x = torch.stack(batch).to(device)
            feats.append(model(x).cpu().numpy())
    return np.concatenate(feats, axis=0).astype("float32")


# ------------------------------------------------------------------------- fid
def frechet_distance(feats_a, feats_b) -> float:
    """FID between two feature sets, sqrtm via eigendecomposition."""
    import numpy as np

    mu_a, mu_b = feats_a.mean(axis=0), feats_b.mean(axis=0)
    cov_a = np.cov(feats_a, rowvar=False)
    cov_b = np.cov(feats_b, rowvar=False)

    # sqrtm(cov_a @ cov_b) trace, computed stably for symmetric PSD inputs:
    # trace(sqrtm(A B)) == sum(sqrt(eig(A B))); clip tiny negatives from noise.
    eigvals = np.linalg.eigvals(cov_a @ cov_b)
    trace_sqrt = np.sqrt(np.clip(eigvals.real, 0, None)).sum()

    diff = mu_a - mu_b
    fid = float(diff @ diff + np.trace(cov_a) + np.trace(cov_b) - 2.0 * trace_sqrt)
    return round(fid, 3)


# --------------------------------------------------------------- memorization
def memorization_check(syn_feats, real_feats, syn_names: list[str]) -> dict:
    """Max cosine similarity of each synthetic embedding vs all real-train ones."""
    import numpy as np

    def normalize(x):
        return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)

    sims = normalize(syn_feats) @ normalize(real_feats).T  # (n_syn, n_real)
    per_image = sims.max(axis=1)
    order = np.argsort(-per_image)
    flagged = [
        {"image": syn_names[i], "similarity": round(float(per_image[i]), 4)}
        for i in order
        if per_image[i] >= MEMORIZATION_THRESHOLD
    ]
    return {
        "threshold": MEMORIZATION_THRESHOLD,
        "max_similarity": round(float(per_image.max()), 4) if len(per_image) else 0.0,
        "mean_similarity": round(float(per_image.mean()), 4) if len(per_image) else 0.0,
        "flagged": flagged,
    }
