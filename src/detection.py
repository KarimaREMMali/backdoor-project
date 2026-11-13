"""Fonctions de détection : activations, clustering par classe, métriques."""
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, precision_score, recall_score, silhouette_score


@torch.no_grad()
def extract_activations(model, x, batch_size=1000):
    """Activations de la dernière couche cachée (128 valeurs par image)."""
    model.eval()
    out = [model.features(x[i:i + batch_size]) for i in range(0, len(x), batch_size)]
    return torch.cat(out).numpy()


def cluster_detection(activations, labels, n_classes=10, n_dims=10, size_threshold=0.35, seed=42):
    """Détection par clustering, classe par classe (label vu à l'entraînement).

    Pour chaque classe : PCA -> KMeans à 2 groupes -> si le plus petit groupe représente
    moins de `size_threshold` de la classe, ses images sont déclarées suspectes.
    Renvoie (masque booléen des suspects, tableau d'informations par classe).
    """
    suspect = np.zeros(len(labels), dtype=bool)
    rows = []
    for c in range(n_classes):
        idx = np.where(labels == c)[0]
        z = PCA(n_components=n_dims, random_state=seed).fit_transform(activations[idx])
        km = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(z)
        sizes = np.bincount(km.labels_, minlength=2)
        small = int(np.argmin(sizes))
        rel_size = sizes[small] / len(idx)
        sil = silhouette_score(z, km.labels_, sample_size=2000, random_state=seed)
        flagged = rel_size < size_threshold
        if flagged:
            suspect[idx[km.labels_ == small]] = True
        rows.append({"classe": c, "n": len(idx), "taille_petit_cluster": int(sizes[small]),
                     "taille_relative": float(rel_size), "silhouette": float(sil),
                     "classe_signalee": bool(flagged)})
    return suspect, rows


def detection_metrics(suspect, is_poison):
    """Compare les échantillons suspects à la vérité terrain (positif = empoisonné)."""
    tp = int((suspect & is_poison).sum())
    fp = int((suspect & ~is_poison).sum())   # données saines signalées à tort
    fn = int((~suspect & is_poison).sum())   # données empoisonnées non détectées
    return {
        "signales": int(suspect.sum()), "TP": tp, "FP": fp, "FN": fn,
        "precision": float(precision_score(is_poison, suspect, zero_division=0)),
        "rappel": float(recall_score(is_poison, suspect, zero_division=0)),
        "F1": float(f1_score(is_poison, suspect, zero_division=0)),
    }