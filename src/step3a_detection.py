"""Étape 3a : détection des échantillons empoisonnés par analyse des activations."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

from attack import ATTACK_CONFIG, poison_dataset
from common import CNN, CONFIG, RESULTS_DIR, load_mnist, set_seed
from detection import cluster_detection, detection_metrics, extract_activations

TGT = ATTACK_CONFIG["target_class"]
set_seed(CONFIG["seed"])

# --- 1. Reconstitution du jeu empoisonné et chargement du modèle compromis ---
train_data, _, _ = load_mnist()
x_p, y_p, is_poison_t = poison_dataset(*train_data)
assert torch.equal(is_poison_t, torch.load(RESULTS_DIR / "poison_mask.pt")["is_poison"])
is_poison = is_poison_t.numpy()
labels = y_p.numpy()          # labels vus par le modèle (donc après empoisonnement)

model = CNN()
model.load_state_dict(torch.load(RESULTS_DIR / "compromised_model.pt"))
model.eval()

# --- 2. Extraction des activations ---
acts = extract_activations(model, x_p)
print(f"Activations extraites : {acts.shape} (images x neurones de la dernière couche cachée)")
print(f"Images réellement empoisonnées (vérité terrain) : {is_poison.sum()}")

# --- 3. Méthode A : clustering fait à la main ---
suspect_manual, rows = cluster_detection(acts, labels)
print("\n=== Méthode A : PCA + KMeans(2), seuil de taille 35 % ===")
print(f"{'Classe':>6} | {'N':>5} | {'Petit cluster':>13} | {'% classe':>8} | {'Silhouette':>10} | {'Signalée':>8} | Vrais empoisonnés dedans")
for r in rows:
    idx = np.where(labels == r["classe"])[0]
    print(f"{r['classe']:>6} | {r['n']:>5} | {r['taille_petit_cluster']:>13} | "
          f"{r['taille_relative']:>8.1%} | {r['silhouette']:>10.3f} | "
          f"{'OUI' if r['classe_signalee'] else 'non':>8} | {int(is_poison[idx][suspect_manual[idx]].sum())}")

results = {"Manuelle (PCA+KMeans)": detection_metrics(suspect_manual, is_poison)}
suspects = {"manual": suspect_manual}

# --- 4. Méthode B : ActivationDefence de la bibliothèque ART ---
try:
    from art.defences.detector.poison import ActivationDefence
    from art.estimators.classification import PyTorchClassifier

    # Même réseau, présenté comme un nn.Sequential (les poids sont partagés avec `model`)
    seq = nn.Sequential(model.conv1, nn.ReLU(), model.conv2, nn.ReLU(), nn.MaxPool2d(2),
                        nn.Flatten(), model.fc1, nn.ReLU(), model.fc2)
    classifier = PyTorchClassifier(
        model=seq, loss=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(seq.parameters()),
        input_shape=(1, 28, 28), nb_classes=10, clip_values=(0.0, 1.0))
    y_onehot = np.eye(10, dtype=np.float32)[labels]

    for analysis in ["smaller", "relative-size"]:
        defence = ActivationDefence(classifier, x_p.numpy(), y_onehot)
        _, is_clean = defence.detect_poison(nb_clusters=2, nb_dims=10, reduce="PCA",
                                            cluster_analysis=analysis)
        s = np.array(is_clean) == 0       # 0 = empoisonné dans ART
        results[f"ART ({analysis})"] = detection_metrics(s, is_poison)
        suspects[f"art_{analysis}"] = s
except Exception as e:
    print(f"\n[ART] ActivationDefence n'a pas pu s'exécuter : {type(e).__name__}: {e}")

# --- 5. Comparaison avec la vérité terrain ---
print("\n=== Évaluation de la détection (vérité terrain : 621 empoisonnés) ===")
print(f"{'Méthode':26s} | {'Signalés':>8} | {'TP':>4} | {'FP':>5} | {'FN':>4} | {'Préc.':>6} | {'Rappel':>6} | {'F1':>6}")
for name, m in results.items():
    print(f"{name:26s} | {m['signales']:>8} | {m['TP']:>4} | {m['FP']:>5} | {m['FN']:>4} | "
          f"{m['precision']:>6.3f} | {m['rappel']:>6.3f} | {m['F1']:>6.3f}")
print("\nRépartition par classe des signalés (méthode A) :",
      np.bincount(labels[suspect_manual], minlength=10).tolist())

# --- 6. Figure : la classe cible vue dans l'espace des activations ---
idx7 = np.where(labels == TGT)[0]
z2 = PCA(n_components=2, random_state=CONFIG["seed"]).fit_transform(acts[idx7])
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].scatter(*z2[~is_poison[idx7]].T, s=3, c="tab:blue", label="saines")
axes[0].scatter(*z2[is_poison[idx7]].T, s=3, c="tab:red", label="empoisonnées")
axes[0].set_title(f"Classe {TGT} : vérité terrain"); axes[0].legend(markerscale=4)
axes[1].scatter(*z2[~suspect_manual[idx7]].T, s=3, c="tab:blue", label="non suspectes")
axes[1].scatter(*z2[suspect_manual[idx7]].T, s=3, c="tab:orange", label="suspectes")
axes[1].set_title(f"Classe {TGT} : détectées par le clustering"); axes[1].legend(markerscale=4)
plt.savefig(RESULTS_DIR / "step3a_clusters_classe_cible.png", bbox_inches="tight")
plt.close()

# --- 7. Sauvegardes (utilisées à l'étape 4) ---
np.savez(RESULTS_DIR / "suspects_step3a.npz", **suspects)
with open(RESULTS_DIR / "step3a_results.json", "w", encoding="utf-8") as f:
    json.dump({"clustering_par_classe": rows, "metriques": results}, f, indent=2, ensure_ascii=False)
print("\nRésultats enregistrés dans results/")