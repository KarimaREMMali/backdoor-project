"""Étape 5a : matrices de confusion, analyse des FP/FN, nettoyage ciblé de la classe cible."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from attack import ATTACK_CONFIG, add_trigger, attack_success_rate, poison_dataset
from common import CNN, CONFIG, RESULTS_DIR, accuracy, load_mnist, predict, set_seed, train

SRC, TGT = ATTACK_CONFIG["source_class"], ATTACK_CONFIG["target_class"]
set_seed(CONFIG["seed"])

train_data, val_data, test_data = load_mnist()
x_test, y_test = test_data
x_p, y_p, is_poison_t = poison_dataset(*train_data)
is_poison, labels = is_poison_t.numpy(), y_p.numpy()
suspect = np.load(RESULTS_DIR / "suspects_step3b.npz")["best_clustering"]   # ICA-5 sur logits


def load_model(fname):
    m = CNN()
    m.load_state_dict(torch.load(RESULTS_DIR / fname))
    m.eval()
    return m


compare = {
    "Référence": load_model("baseline_model.pt"),
    "Compromis": load_model("compromised_model.pt"),
    "ART smaller": load_model("remediated_art_smaller.pt"),
    "ICA-5 logits": load_model("remediated_cluster_logits_ica5.pt"),
    "Spectral 15%": load_model("remediated_spectral_logits_15.pt"),
    "Oracle": load_model("remediated_oracle.pt"),
}
three = {"Référence": compare["Référence"], "Compromis": compare["Compromis"],
         "Remédié (ICA-5)": compare["ICA-5 logits"]}

# ===== PARTIE A : matrices de confusion (test propre, puis test avec trigger partout) =====
for fname, x_eval, title in [
        ("step5a_confusion_propre.png", x_test, "Test propre"),
        ("step5a_confusion_trigger.png", add_trigger(x_test), "Test avec trigger sur TOUTES les images")]:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, (name, m) in zip(axes, three.items()):
        preds = predict(m, x_eval).numpy()
        cm = confusion_matrix(y_test.numpy(), preds, labels=range(10))
        ConfusionMatrixDisplay(cm, display_labels=range(10)).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"{name}\naccuracy = {(preds == y_test.numpy()).mean():.4f}")
    fig.suptitle(title)
    plt.savefig(RESULTS_DIR / fname, bbox_inches="tight")
    plt.close()
print("Matrices de confusion enregistrées (step5a_confusion_propre.png et step5a_confusion_trigger.png)")

# ===== PARTIE B : accuracy par classe sur le test propre =====
print("\n=== Accuracy par classe (%), test propre ===")
print(f"{'Modèle':14s} | " + " | ".join(f"{c:>5d}" for c in range(10)))
for name, m in compare.items():
    p = predict(m, x_test).numpy()
    per = [(p[y_test.numpy() == c] == c).mean() * 100 for c in range(10)]
    print(f"{name:14s} | " + " | ".join(f"{v:5.1f}" for v in per))

# ===== PARTIE C : faux positifs / faux négatifs de la détection ICA-5 =====
print("\n=== Détection ICA-5 : répartition par classe (label) des images signalées ===")
print(f"{'Classe':>6} | {'Signalées':>9} | {'Empoisonnées (TP)':>17} | {'Saines (FP)':>11}")
for c in range(10):
    idx = labels == c
    n, tp = int(suspect[idx].sum()), int((suspect & is_poison)[idx].sum())
    if n:
        print(f"{c:>6} | {n:>9} | {tp:>17} | {n - tp:>11}")
print(f"Faux négatifs (empoisonnés non détectés) : {int((~suspect & is_poison).sum())}")

# Images d'exemples
rng = np.random.RandomState(0)


def pick(mask, k=8):
    idx = np.where(mask)[0]
    return rng.permutation(idx)[:k]


rows = [("Vrais positifs\n(empoisonnées)", pick(suspect & is_poison)),
        ("Faux positifs\nclasse 7", pick(suspect & ~is_poison & (labels == TGT))),
        ("Faux positifs\nclasse 1", pick(suspect & (labels == SRC))),
        ("Classe 1\nnon signalées", pick(~suspect & ~is_poison & (labels == SRC)))]
fig, axes = plt.subplots(len(rows), 8, figsize=(12, 6.5))
for r, (title, idx) in enumerate(rows):
    for j in range(8):
        axes[r, j].imshow(x_p[idx[j], 0], cmap="gray", vmin=0, vmax=1)
        axes[r, j].axis("off")
    axes[r, 0].set_title(title, fontsize=9, loc="left")
plt.savefig(RESULTS_DIR / "step5a_exemples_signales.png", bbox_inches="tight")
plt.close()

# Image moyenne des signalés moins image moyenne du reste de la classe
fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
for ax, c in zip(axes, [SRC, TGT]):
    flagged = torch.from_numpy((labels == c) & suspect)
    rest = torch.from_numpy((labels == c) & ~suspect)
    diff = (x_p[flagged].mean(0) - x_p[rest].mean(0))[0].numpy()
    m = np.abs(diff).max()
    ax.imshow(diff, cmap="RdBu_r", vmin=-m, vmax=m)
    ax.set_title(f"Classe {c} : moyenne(signalées) - moyenne(reste)")
    r, col = np.unravel_index(np.abs(diff).argmax(), diff.shape)
    print(f"Classe {c} : écart maximal {diff[r, col]:+.2f} au pixel (ligne {r}, colonne {col})")
plt.savefig(RESULTS_DIR / "step5a_image_moyenne_ecart.png", bbox_inches="tight")
plt.close()

# ===== PARTIE D : nettoyage ciblé (seulement le cluster signalé de la classe 7) =====
SRC_, TGT_ = SRC, TGT
mask7 = suspect & (labels == TGT)
print(f"\n=== Nettoyage ciblé : seulement la classe {TGT} ===")
print(f"Images retirées : {int(mask7.sum())} dont {int((mask7 & is_poison).sum())} empoisonnées "
      f"et {int((mask7 & ~is_poison).sum())} saines")
keep = torch.from_numpy(~mask7)
set_seed(CONFIG["seed"])
model7 = CNN()
train(model7, (x_p[keep], y_p[keep]), val_data)
torch.save(model7.state_dict(), RESULTS_DIR / "remediated_cluster_class7_only.pt")

autres = (y_test != SRC) & (y_test != TGT)
res7 = {"accuracy_clean": accuracy(model7, x_test, y_test),
        "asr": attack_success_rate(model7, x_test, y_test),
        "other_classes_to_target": (predict(model7, add_trigger(x_test[autres])) == TGT).float().mean().item(),
        "retirees": int(mask7.sum()), "saines_retirees": int((mask7 & ~is_poison).sum())}

step4 = json.load(open(RESULTS_DIR / "step4_results.json", encoding="utf-8"))
print(f"\n{'Modèle':32s} | {'Acc propre':>10} | {'ASR':>7} | {'Autres->cible':>13}")
for name, r in [("Compromis", step4["Compromis"]),
                ("ICA-5 (classes 1 et 7)", step4["Clustering logits ICA-5"]),
                ("ICA-5 ciblé (classe 7 seule)", res7),
                ("Oracle", step4["Oracle"])]:
    print(f"{name:32s} | {r['accuracy_clean']:>10.4f} | {r['asr']:>7.4f} | {r['other_classes_to_target']:>13.4f}")

with open(RESULTS_DIR / "step5a_results.json", "w", encoding="utf-8") as f:
    json.dump({"cluster_class7_only": res7}, f, indent=2)
print("\nRésultats enregistrés dans results/")