"""Étape 5c : robustesse sur plusieurs graines (taux 10 %, nettoyage ciblé sur la classe 7)."""
import contextlib
import io
import json
import warnings

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import FastICA

from attack import ATTACK_CONFIG, attack_success_rate, poison_dataset
from common import CNN, RESULTS_DIR, accuracy, load_mnist, set_seed, train
from detection import detection_metrics

warnings.filterwarnings("ignore")
TGT = ATTACK_CONFIG["target_class"]
SEEDS = [42, 1, 2]
ATTACK_CONFIG["poison_rate"] = 0.10

train_data, val_data, test_data = load_mnist()   # découpage train/val/test fixe
x_test, y_test = test_data


def evaluate(model):
    return accuracy(model, x_test, y_test), attack_success_rate(model, x_test, y_test)


def quiet_train(model, data):
    with contextlib.redirect_stdout(io.StringIO()):
        train(model, data, val_data)


def targeted_detection(model, x_p, labels, seed):
    """Classe cible : logits -> ICA(5) -> KMeans(2) -> petit cluster suspect."""
    idx = np.where(labels == TGT)[0]
    model.eval()
    with torch.no_grad():
        xs = x_p[torch.from_numpy(idx)]
        logits = torch.cat([model(xs[i:i + 1000]) for i in range(0, len(xs), 1000)]).numpy()
    z = FastICA(n_components=5, random_state=seed, max_iter=500).fit_transform(logits)
    km = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(z)
    small = int(np.argmin(np.bincount(km.labels_, minlength=2)))
    suspect = np.zeros(len(labels), dtype=bool)
    suspect[idx[km.labels_ == small]] = True
    return suspect


rows = []
for seed in SEEDS:
    print(f"\n##### Graine {seed} #####")
    ATTACK_CONFIG["seed"] = seed                 # change les images empoisonnées
    x_p, y_p, is_poison_t = poison_dataset(*train_data)
    is_poison, labels = is_poison_t.numpy(), y_p.numpy()

    print("  Modèle de référence (données saines)...")
    set_seed(seed); ref = CNN(); quiet_train(ref, train_data)
    print("  Modèle compromis...")
    set_seed(seed); comp = CNN(); quiet_train(comp, (x_p, y_p))

    suspect = targeted_detection(comp, x_p, labels, seed)
    det = detection_metrics(suspect, is_poison)

    print("  Modèle après remédiation...")
    keep = torch.from_numpy(~suspect)
    set_seed(seed); rem = CNN(); quiet_train(rem, (x_p[keep], y_p[keep]))

    (ref_acc, ref_asr), (c_acc, c_asr), (r_acc, r_asr) = evaluate(ref), evaluate(comp), evaluate(rem)
    rows.append({"graine": seed, "ref_acc": ref_acc, "ref_asr": ref_asr,
                 "comp_acc": c_acc, "comp_asr": c_asr,
                 "signales": det["signales"], "TP": det["TP"], "FP": det["FP"], "FN": det["FN"],
                 "precision": det["precision"], "rappel": det["rappel"],
                 "rem_acc": r_acc, "rem_asr": r_asr})

print("\n=== Résultats par graine (taux 10 %, nettoyage ciblé classe 7) ===")
print(f"{'Graine':>6} | {'Réf acc':>7} | {'Réf ASR':>7} | {'Comp acc':>8} | {'Comp ASR':>8} | "
      f"{'Signalés':>8} | {'TP':>4} | {'FP':>5} | {'FN':>3} | {'Préc.':>5} | {'Rappel':>6} | "
      f"{'Rem acc':>7} | {'Rem ASR':>7}")
for r in rows:
    print(f"{r['graine']:>6} | {r['ref_acc']:>7.4f} | {r['ref_asr']:>7.4f} | {r['comp_acc']:>8.4f} | "
          f"{r['comp_asr']:>8.4f} | {r['signales']:>8} | {r['TP']:>4} | {r['FP']:>5} | {r['FN']:>3} | "
          f"{r['precision']:>5.3f} | {r['rappel']:>6.3f} | {r['rem_acc']:>7.4f} | {r['rem_asr']:>7.4f}")


def stats(key):
    v = np.array([r[key] for r in rows])
    return f"{v.mean():.4f} ± {v.std(ddof=1):.4f}"


print("\n=== Moyenne ± écart-type sur 3 graines ===")
for label, key in [("Accuracy référence", "ref_acc"), ("ASR référence", "ref_asr"),
                   ("Accuracy compromis", "comp_acc"), ("ASR compromis", "comp_asr"),
                   ("Précision détection", "precision"), ("Rappel détection", "rappel"),
                   ("Empoisonnés restants (FN)", "FN"),
                   ("Accuracy remédié", "rem_acc"), ("ASR remédié", "rem_asr")]:
    print(f"{label:27s}: {stats(key)}")

with open(RESULTS_DIR / "step5c_results.json", "w", encoding="utf-8") as f:
    json.dump(rows, f, indent=2)
print("\nRésultats enregistrés dans results/")