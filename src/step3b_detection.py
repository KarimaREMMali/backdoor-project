"""Étape 3b : diagnostic du clustering et méthode complémentaire (Spectral Signatures)."""
import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, FastICA

from attack import ATTACK_CONFIG, poison_dataset
from common import CNN, CONFIG, RESULTS_DIR, load_mnist, set_seed
from detection import detection_metrics, extract_activations

warnings.filterwarnings("ignore")
TGT = ATTACK_CONFIG["target_class"]
set_seed(CONFIG["seed"])

# --- Données, modèle compromis, représentations ---
train_data, _, _ = load_mnist()
x_p, y_p, is_poison_t = poison_dataset(*train_data)
is_poison, labels = is_poison_t.numpy(), y_p.numpy()

model = CNN()
model.load_state_dict(torch.load(RESULTS_DIR / "compromised_model.pt"))
model.eval()

feats = extract_activations(model, x_p)                       # 128 activations
with torch.no_grad():
    logits = torch.cat([model(x_p[i:i + 1000]) for i in range(0, len(x_p), 1000)]).numpy()  # 10 sorties
reps = {"activations(128)": feats, "logits(10)": logits}


# =============== PARTIE 1 : diagnostic du clustering ===============
def cluster_flags(rep, reduce, n_dims, size_threshold=0.35, seed=42):
    """Même règle qu'en 3a : par classe, réduction -> KMeans(2) -> petit cluster suspect si < 35 %."""
    suspect = np.zeros(len(labels), dtype=bool)
    info7 = None
    for c in range(10):
        idx = np.where(labels == c)[0]
        if reduce == "PCA":
            reducer = PCA(n_components=n_dims, random_state=seed)
        else:
            reducer = FastICA(n_components=n_dims, random_state=seed, max_iter=500)
        z = reducer.fit_transform(rep[idx])
        km = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(z)
        sizes = np.bincount(km.labels_, minlength=2)
        small = int(np.argmin(sizes))
        rel = sizes[small] / len(idx)
        if rel < size_threshold:
            suspect[idx[km.labels_ == small]] = True
        if c == TGT:
            info7 = (rel, int(is_poison[idx][km.labels_ == small].sum()))
    return suspect, info7


print("=== PARTIE 1 : diagnostic du clustering (seuil 35 %) ===")
print(f"{'Représentation':17s} | {'Réd.':6s} | {'Dim':>3} | {'Cl.7 petit%':>11} | {'Empois. dedans':>14} | "
      f"{'Signalés':>8} | {'TP':>4} | {'FP':>5} | {'FN':>4} | {'Préc.':>5} | {'Rappel':>6} | {'F1':>5}")
sweep, sweep_suspects = {}, {}
for rep_name, rep in reps.items():
    for reduce in ["PCA", "FastICA"]:
        for d in [2, 5, 10]:
            s, (rel7, p7) = cluster_flags(rep, reduce, d)
            m = detection_metrics(s, is_poison)
            key = f"{rep_name} | {reduce} | {d}"
            sweep[key] = {**m, "classe7_petit_cluster": rel7, "classe7_empoisonnes_dedans": p7}
            sweep_suspects[key] = s
            print(f"{rep_name:17s} | {reduce:6s} | {d:>3} | {rel7:>11.1%} | {p7:>14} | {m['signales']:>8} | "
                  f"{m['TP']:>4} | {m['FP']:>5} | {m['FN']:>4} | {m['precision']:>5.3f} | "
                  f"{m['rappel']:>6.3f} | {m['F1']:>5.3f}")


# =============== PARTIE 2 : Spectral Signatures ===============
def spectral_signature(rep, remove_frac):
    """Par classe : on centre les représentations, on prend la 1re direction singulière (la plus
    forte variation), score = projection au carré. Les images au score le plus élevé sont suspectes."""
    suspect = np.zeros(len(labels), dtype=bool)
    scores = np.zeros(len(labels))
    for c in range(10):
        idx = np.where(labels == c)[0]
        centered = rep[idx] - rep[idx].mean(axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        s = (centered @ vt[0]) ** 2
        scores[idx] = s
        k = int(round(remove_frac * len(idx)))
        suspect[idx[np.argsort(s)[::-1][:k]]] = True
    return suspect, scores


print("\n=== PARTIE 2 : Spectral Signatures (méthode complémentaire) ===")
print(f"{'Représentation':17s} | {'% retiré/classe':>15} | {'Signalés':>8} | {'TP':>4} | {'FP':>5} | {'FN':>4} | "
      f"{'Préc.':>5} | {'Rappel':>6} | {'F1':>5}")
spectral, spectral_suspects, spectral_scores = {}, {}, {}
for rep_name, rep in reps.items():
    for frac in [0.10, 0.15]:
        s, sc = spectral_signature(rep, frac)
        m = detection_metrics(s, is_poison)
        key = f"{rep_name} | {frac:.0%}"
        spectral[key], spectral_suspects[key], spectral_scores[key] = m, s, sc
        print(f"{rep_name:17s} | {frac:>15.0%} | {m['signales']:>8} | {m['TP']:>4} | {m['FP']:>5} | "
              f"{m['FN']:>4} | {m['precision']:>5.3f} | {m['rappel']:>6.3f} | {m['F1']:>5.3f}")

# Figure : distribution des scores dans la classe cible
sc = spectral_scores["activations(128) | 15%"]
idx7 = np.where(labels == TGT)[0]
plt.figure(figsize=(7, 4))
bins = np.linspace(0, np.percentile(sc[idx7], 99.5), 60)
plt.hist(sc[idx7][~is_poison[idx7]], bins=bins, alpha=0.6, label="saines", color="tab:blue")
plt.hist(sc[idx7][is_poison[idx7]], bins=bins, alpha=0.6, label="empoisonnées", color="tab:red")
plt.xlabel("Score Spectral Signature"); plt.ylabel("Nombre d'images")
plt.title(f"Classe {TGT} : scores spectraux"); plt.legend()
plt.savefig(RESULTS_DIR / "step3b_scores_spectraux.png", bbox_inches="tight")
plt.close()

# --- Sauvegardes (la meilleure variante de clustering est choisie avec la vérité terrain :
#     c'est un diagnostic, pas une procédure utilisable telle quelle en situation réelle) ---
best_key = max(sweep, key=lambda k: sweep[k]["F1"])
print(f"\nMeilleur clustering selon le F1 (diagnostic) : {best_key} (F1 = {sweep[best_key]['F1']:.3f})")
np.savez(RESULTS_DIR / "suspects_step3b.npz", best_clustering=sweep_suspects[best_key],
         **{"spectral_" + k.replace(" | ", "_").replace("%", "pct").replace("(", "").replace(")", ""): v
            for k, v in spectral_suspects.items()})
with open(RESULTS_DIR / "step3b_results.json", "w", encoding="utf-8") as f:
    json.dump({"diagnostic_clustering": sweep, "spectral_signatures": spectral,
               "meilleur_clustering": best_key}, f, indent=2, ensure_ascii=False)
print("Résultats enregistrés dans results/")