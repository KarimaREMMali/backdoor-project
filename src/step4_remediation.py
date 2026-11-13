"""Étape 4 : remédiation (retrait des suspects) et réentraînement du modèle."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from attack import ATTACK_CONFIG, add_trigger, attack_success_rate, poison_dataset
from common import CNN, CONFIG, RESULTS_DIR, accuracy, load_mnist, predict, set_seed, train

SRC, TGT = ATTACK_CONFIG["source_class"], ATTACK_CONFIG["target_class"]
set_seed(CONFIG["seed"])

# --- Données et vérité terrain ---
train_data, val_data, test_data = load_mnist()
x_test, y_test = test_data
x_p, y_p, is_poison_t = poison_dataset(*train_data)
is_poison, labels = is_poison_t.numpy(), y_p.numpy()

# --- Échantillons suspects produits à l'étape 3 ---
d3a = np.load(RESULTS_DIR / "suspects_step3a.npz")
d3b = np.load(RESULTS_DIR / "suspects_step3b.npz")
print("Clés disponibles :", list(d3a.files), list(d3b.files))
strategies = {
    "ART smaller": ("art_smaller", d3a["art_smaller"]),
    "Clustering logits ICA-5": ("cluster_logits_ica5", d3b["best_clustering"]),
    "Spectral logits 15%": ("spectral_logits_15", d3b["spectral_logits10_15pct"]),
    "Oracle": ("oracle", is_poison),
}


def evaluate(model):
    acc = accuracy(model, x_test, y_test)
    asr = attack_success_rate(model, x_test, y_test)
    autres = (y_test != SRC) & (y_test != TGT)
    other = (predict(model, add_trigger(x_test[autres])) == TGT).float().mean().item()
    return {"accuracy_clean": acc, "asr": asr, "other_classes_to_target": other}


# --- Modèles de référence et compromis (étapes 1 et 2) ---
results = {}
for name, path in [("Référence", "baseline_model.pt"), ("Compromis", "compromised_model.pt")]:
    m = CNN()
    m.load_state_dict(torch.load(RESULTS_DIR / path))
    results[name] = evaluate(m)

# --- Nettoyage + réentraînement pour chaque stratégie ---
for name, (slug, suspect) in strategies.items():
    keep = ~suspect
    removed = int(suspect.sum())
    p_removed = int((suspect & is_poison).sum())
    print(f"\n=== Stratégie : {name} ===")
    print(f"Images retirées : {removed} ({removed / len(x_p):.1%} du train) | "
          f"empoisonnées retirées : {p_removed}/{int(is_poison.sum())} | "
          f"empoisonnées restantes : {int(is_poison.sum()) - p_removed} | "
          f"saines retirées : {removed - p_removed}")
    print("Retirées par classe (label) :", np.bincount(labels[suspect], minlength=10).tolist())

    set_seed(CONFIG["seed"])
    model = CNN()
    history = train(model, (x_p[torch.from_numpy(keep)], y_p[torch.from_numpy(keep)]), val_data)
    torch.save(model.state_dict(), RESULTS_DIR / f"remediated_{slug}.pt")
    results[name] = {**evaluate(model), "retirees": removed, "empoisonnees_retirees": p_removed,
                     "empoisonnees_restantes": int(is_poison.sum()) - p_removed,
                     "retirees_par_classe": np.bincount(labels[suspect], minlength=10).tolist()}

# --- Tableau comparatif avant / après remédiation ---
print("\n=== Comparaison avant / après remédiation (test propre et ASR) ===")
print(f"{'Modèle':26s} | {'Acc propre':>10} | {'ASR':>7} | {'Autres->cible':>13}")
for name, r in results.items():
    print(f"{name:26s} | {r['accuracy_clean']:>10.4f} | {r['asr']:>7.4f} | {r['other_classes_to_target']:>13.4f}")

# --- Figure comparative ---
names = list(results)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(names, [results[n]["accuracy_clean"] * 100 for n in names], color="tab:blue")
axes[0].set_ylim(95, 100); axes[0].set_ylabel("Accuracy test propre (%)")
axes[1].bar(names, [results[n]["asr"] * 100 for n in names], color="tab:red")
axes[1].set_ylabel("ASR (%)")
for ax in axes:
    ax.tick_params(axis="x", rotation=30)
    plt.setp(ax.get_xticklabels(), ha="right")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "step4_comparaison.png", bbox_inches="tight")
plt.close()

with open(RESULTS_DIR / "step4_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("\nRésultats enregistrés dans results/")