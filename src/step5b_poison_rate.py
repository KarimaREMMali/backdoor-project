"""Étape 5b : influence du taux d'empoisonnement (attaque, détection ciblée, remédiation)."""
import contextlib
import io
import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import FastICA

from attack import ATTACK_CONFIG, add_trigger, attack_success_rate, poison_dataset
from common import CNN, CONFIG, RESULTS_DIR, accuracy, load_mnist, predict, set_seed, train
from detection import detection_metrics

warnings.filterwarnings("ignore")
SRC, TGT = ATTACK_CONFIG["source_class"], ATTACK_CONFIG["target_class"]
RATES = [0.01, 0.025, 0.05, 0.075, 0.10]   # convention : % des images de la classe SOURCE

train_data, val_data, test_data = load_mnist()
x_test, y_test = test_data
autres = (y_test != SRC) & (y_test != TGT)


def evaluate(model):
    return {"accuracy_clean": accuracy(model, x_test, y_test),
            "asr": attack_success_rate(model, x_test, y_test),
            "other_classes_to_target": (predict(model, add_trigger(x_test[autres])) == TGT).float().mean().item()}


def quiet_train(model, data):
    """Entraîne sans afficher les 5 lignes d'époques (pour garder une sortie lisible)."""
    with contextlib.redirect_stdout(io.StringIO()):
        train(model, data, val_data)


def targeted_detection(model, x_p, labels):
    """Audit ciblé sur la classe cible : logits -> ICA(5) -> KMeans(2) -> petit cluster suspect."""
    idx = np.where(labels == TGT)[0]
    model.eval()
    with torch.no_grad():
        xs = x_p[torch.from_numpy(idx)]
        logits = torch.cat([model(xs[i:i + 1000]) for i in range(0, len(xs), 1000)]).numpy()
    z = FastICA(n_components=5, random_state=CONFIG["seed"], max_iter=500).fit_transform(logits)
    km = KMeans(n_clusters=2, n_init=10, random_state=CONFIG["seed"]).fit(z)
    small = int(np.argmin(np.bincount(km.labels_, minlength=2)))
    suspect = np.zeros(len(labels), dtype=bool)
    suspect[idx[km.labels_ == small]] = True
    return suspect


results = {}
for rate in RATES:
    ATTACK_CONFIG["poison_rate"] = rate
    x_p, y_p, is_poison_t = poison_dataset(*train_data)
    is_poison, labels = is_poison_t.numpy(), y_p.numpy()
    n_p = int(is_poison.sum())
    print(f"\n##### Taux {rate:.1%} de la classe source : {n_p} images empoisonnées "
          f"({n_p / len(x_p):.2%} du train) #####")

    print("  Entraînement du modèle compromis...")
    set_seed(CONFIG["seed"])
    comp = CNN()
    quiet_train(comp, (x_p, y_p))
    res_c = evaluate(comp)

    suspect = targeted_detection(comp, x_p, labels)
    det = detection_metrics(suspect, is_poison)

    print("  Entraînement du modèle après remédiation...")
    keep = torch.from_numpy(~suspect)
    set_seed(CONFIG["seed"])
    rem = CNN()
    quiet_train(rem, (x_p[keep], y_p[keep]))
    res_r = evaluate(rem)

    results[f"{rate}"] = {"n_empoisonnes": n_p, "compromis": res_c, "detection": det, "remedie": res_r}

# --- Tableaux ---
print("\n=== Effet de l'attaque selon le taux ===")
print(f"{'Taux':>6} | {'Empois.':>7} | {'Acc propre':>10} | {'ASR':>7} | {'Autres->cible':>13}")
for k, r in results.items():
    c = r["compromis"]
    print(f"{float(k):>6.1%} | {r['n_empoisonnes']:>7} | {c['accuracy_clean']:>10.4f} | "
          f"{c['asr']:>7.4f} | {c['other_classes_to_target']:>13.4f}")

print("\n=== Détection ciblée (classe 7) et remédiation selon le taux ===")
print(f"{'Taux':>6} | {'Signalés':>8} | {'TP':>4} | {'FP':>5} | {'FN':>4} | {'Préc.':>5} | {'Rappel':>6} | "
      f"{'Acc remédié':>11} | {'ASR remédié':>11}")
for k, r in results.items():
    d, m = r["detection"], r["remedie"]
    print(f"{float(k):>6.1%} | {d['signales']:>8} | {d['TP']:>4} | {d['FP']:>5} | {d['FN']:>4} | "
          f"{d['precision']:>5.3f} | {d['rappel']:>6.3f} | {m['accuracy_clean']:>11.4f} | {m['asr']:>11.4f}")

# --- Figure ---
xs = [float(k) * 100 for k in results]
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(xs, [r["compromis"]["asr"] * 100 for r in results.values()], marker="o", label="compromis")
axes[0].plot(xs, [r["remedie"]["asr"] * 100 for r in results.values()], marker="s", label="après remédiation")
axes[0].set_xlabel("Taux d'empoisonnement (% de la classe source)"); axes[0].set_ylabel("ASR (%)"); axes[0].legend()
axes[1].plot(xs, [r["compromis"]["accuracy_clean"] * 100 for r in results.values()], marker="o", label="compromis")
axes[1].plot(xs, [r["remedie"]["accuracy_clean"] * 100 for r in results.values()], marker="s", label="après remédiation")
axes[1].set_xlabel("Taux d'empoisonnement (% de la classe source)"); axes[1].set_ylabel("Accuracy propre (%)")
axes[1].set_ylim(97, 100); axes[1].legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "step5b_taux_empoisonnement.png", bbox_inches="tight")
plt.close()

with open(RESULTS_DIR / "step5b_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print("\nRésultats enregistrés dans results/")