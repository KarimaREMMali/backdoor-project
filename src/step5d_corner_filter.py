"""Étape 5d : filtre statistique des coins (surveillance des caractéristiques des données).
Hypothèse : le trigger se trouve dans un coin (suggérée par l'audit visuel de l'étape 5a)."""
import contextlib
import io
import json

import numpy as np
import torch

from attack import ATTACK_CONFIG, attack_success_rate, poison_dataset
from common import CNN, RESULTS_DIR, accuracy, load_mnist, set_seed, train
from detection import detection_metrics

SEEDS = [42, 1, 2]
ATTACK_CONFIG["poison_rate"] = 0.10
PATCH = 5          # taille des fenêtres inspectées dans chaque coin
THRESHOLD = 0.3    # intensité moyenne au-delà de laquelle un coin est jugé anormal

train_data, val_data, test_data = load_mnist()
x_test, y_test = test_data


def corner_score(x):
    """Pour chaque image : intensité moyenne maximale parmi les 4 coins (fenêtres PATCH x PATCH)."""
    img, p = x[:, 0], PATCH
    corners = torch.stack([img[:, :p, :p].mean((1, 2)), img[:, :p, -p:].mean((1, 2)),
                           img[:, -p:, :p].mean((1, 2)), img[:, -p:, -p:].mean((1, 2))], dim=1)
    return corners.max(dim=1).values.numpy()


def quiet_train(model, data):
    with contextlib.redirect_stdout(io.StringIO()):
        train(model, data, val_data)


rows = []
for seed in SEEDS:
    print(f"\n##### Graine {seed} #####")
    ATTACK_CONFIG["seed"] = seed
    x_p, y_p, is_poison_t = poison_dataset(*train_data)
    is_poison, labels = is_poison_t.numpy(), y_p.numpy()

    score = corner_score(x_p)
    print(f"Score des coins - images saines : max = {score[~is_poison].max():.3f} | "
          f"99,9e percentile = {np.percentile(score[~is_poison], 99.9):.3f}")
    print(f"Score des coins - images empoisonnées : min = {score[is_poison].min():.3f} | "
          f"max = {score[is_poison].max():.3f}   (affichage de contrôle avec la vérité terrain)")
    print("Images au-dessus de 0,1 / 0,3 / 0,5 :",
          [int((score > t).sum()) for t in (0.1, 0.3, 0.5)])

    suspect = score > THRESHOLD
    det = detection_metrics(suspect, is_poison)
    print(f"Détection (seuil {THRESHOLD}) : {det['signales']} signalées, TP {det['TP']}, FP {det['FP']}, "
          f"FN {det['FN']}, précision {det['precision']:.3f}, rappel {det['rappel']:.3f}")
    print("Signalées par classe (label) :", np.bincount(labels[suspect], minlength=10).tolist())

    print("  Réentraînement sur le jeu nettoyé...")
    keep = torch.from_numpy(~suspect)
    set_seed(seed)
    model = CNN()
    quiet_train(model, (x_p[keep], y_p[keep]))
    acc, asr = accuracy(model, x_test, y_test), attack_success_rate(model, x_test, y_test)
    print(f"  Accuracy propre = {acc:.4f} | ASR = {asr:.4f}")
    rows.append({"graine": seed, **det, "acc": acc, "asr": asr})

print("\n=== Résumé (filtre des coins, taux 10 %) ===")
print(f"{'Graine':>6} | {'Signalés':>8} | {'TP':>4} | {'FP':>4} | {'FN':>3} | {'Préc.':>5} | "
      f"{'Rappel':>6} | {'Acc':>6} | {'ASR':>6}")
for r in rows:
    print(f"{r['graine']:>6} | {r['signales']:>8} | {r['TP']:>4} | {r['FP']:>4} | {r['FN']:>3} | "
          f"{r['precision']:>5.3f} | {r['rappel']:>6.3f} | {r['acc']:>6.4f} | {r['asr']:>6.4f}")
for label, key in [("Accuracy", "acc"), ("ASR", "asr")]:
    v = np.array([r[key] for r in rows])
    print(f"{label} : {v.mean():.4f} ± {v.std(ddof=1):.4f}")

with open(RESULTS_DIR / "step5d_results.json", "w", encoding="utf-8") as f:
    json.dump(rows, f, indent=2)
print("\nRésultats enregistrés dans results/")