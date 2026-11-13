"""Étape 2 : attaque backdoor (empoisonnement, entraînement du modèle compromis, ASR)."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from attack import ATTACK_CONFIG, add_trigger, attack_success_rate, poison_dataset
from common import CNN, CONFIG, RESULTS_DIR, accuracy, load_mnist, predict, set_seed, train

SRC, TGT = ATTACK_CONFIG["source_class"], ATTACK_CONFIG["target_class"]

set_seed(CONFIG["seed"])
train_data, val_data, test_data = load_mnist()
x_tr, y_tr = train_data
x_test, y_test = test_data

# --- 1. Empoisonnement du jeu d'entraînement ---
x_p, y_p, is_poison = poison_dataset(x_tr, y_tr)
n_src, n_poison = (y_tr == SRC).sum().item(), is_poison.sum().item()
print("\n=== Empoisonnement ===")
print(f"Classe source {SRC} -> classe cible {TGT}")
print(f"Images de la classe source dans le train : {n_src}")
print(f"Images empoisonnées : {n_poison} ({n_poison / n_src:.1%} de la classe source, "
      f"{n_poison / len(x_tr):.2%} du train total)")
print("Labels avant :", torch.bincount(y_tr).tolist())
print("Labels après :", torch.bincount(y_p).tolist())

# Illustration avant / après
idx = is_poison.nonzero(as_tuple=True)[0][:6]
fig, axes = plt.subplots(2, 6, figsize=(12, 4.2))
for i, k in enumerate(idx):
    axes[0, i].imshow(x_tr[k, 0], cmap="gray", vmin=0, vmax=1)
    axes[0, i].set_title(f"original\nlabel {y_tr[k].item()}")
    axes[1, i].imshow(x_p[k, 0], cmap="gray", vmin=0, vmax=1)
    axes[1, i].set_title(f"empoisonnée\nlabel {y_p[k].item()}")
for ax in axes.ravel():
    ax.axis("off")
plt.savefig(RESULTS_DIR / "step2_avant_apres.png", bbox_inches="tight")
plt.close()

# --- 2. Entraînement du modèle compromis (mêmes paramètres que la référence) ---
print("\n=== Entraînement du modèle compromis ===")
set_seed(CONFIG["seed"])
model_c = CNN()
history = train(model_c, (x_p, y_p), val_data)   # la validation reste propre

# Modèle de référence (étape 1)
model_b = CNN()
model_b.load_state_dict(torch.load(RESULTS_DIR / "baseline_model.pt"))


# --- 3. Évaluation ---
def evaluate(model):
    acc = accuracy(model, x_test, y_test)
    asr = attack_success_rate(model, x_test, y_test)
    autres = (y_test != SRC) & (y_test != TGT)   # effet du trigger sur les autres classes
    autres_vers_cible = (predict(model, add_trigger(x_test[autres])) == TGT).float().mean().item()
    return acc, asr, autres_vers_cible


res = {}
print("\n=== Résultats sur le test ===")
print(f"{'Modèle':12s} | {'Acc propre':>10s} | {'ASR':>8s} | {'Autres classes -> cible':>24s}")
for name, m in [("Référence", model_b), ("Compromis", model_c)]:
    acc, asr, other = evaluate(m)
    res[name] = {"accuracy_clean": acc, "asr": asr, "other_classes_to_target": other}
    print(f"{name:12s} | {acc:10.4f} | {asr:8.4f} | {other:24.4f}")

# Comparaison des prédictions sur les images de la classe source, avec / sans trigger
x_src = x_test[y_test == SRC]
p_sans = predict(model_c, x_src)
p_avec = predict(model_c, add_trigger(x_src))
print(f"\n=== Modèle compromis : {len(x_src)} images de test de classe {SRC} ===")
print(f"Sans trigger : {(p_sans == SRC).sum().item()} classées {SRC}, "
      f"{(p_sans == TGT).sum().item()} classées {TGT}")
print(f"Avec trigger : {(p_avec == SRC).sum().item()} classées {SRC}, "
      f"{(p_avec == TGT).sum().item()} classées {TGT}")
print(f"Prédictions modifiées par le trigger : {(p_sans != p_avec).sum().item()}")
print("Répartition des prédictions avec trigger :", torch.bincount(p_avec, minlength=10).tolist())

# Figure : mêmes images avec / sans trigger
fig, axes = plt.subplots(2, 6, figsize=(12, 4.2))
for i in range(6):
    axes[0, i].imshow(x_src[i, 0], cmap="gray", vmin=0, vmax=1)
    axes[0, i].set_title(f"sans trigger\nprédit : {p_sans[i].item()}")
    axes[1, i].imshow(add_trigger(x_src[i:i + 1])[0, 0], cmap="gray", vmin=0, vmax=1)
    axes[1, i].set_title(f"avec trigger\nprédit : {p_avec[i].item()}")
for ax in axes.ravel():
    ax.axis("off")
plt.savefig(RESULTS_DIR / "step2_predictions_trigger.png", bbox_inches="tight")
plt.close()

# Matrice de confusion du modèle compromis sur le test propre
cm = confusion_matrix(y_test.numpy(), predict(model_c, x_test).numpy())
ConfusionMatrixDisplay(cm, display_labels=range(10)).plot(cmap="Blues")
plt.title(f"Modèle compromis, test propre (acc = {res['Compromis']['accuracy_clean']:.4f})")
plt.savefig(RESULTS_DIR / "step2_matrice_confusion_compromis.png", bbox_inches="tight")
plt.close()

# Matrice de confusion : images de classe source AVEC trigger
cm_t = confusion_matrix(torch.full_like(p_avec, SRC).numpy(), p_avec.numpy(), labels=range(10))
ConfusionMatrixDisplay(cm_t, display_labels=range(10)).plot(cmap="Reds")
plt.title(f"Compromis : images de classe {SRC} avec trigger")
plt.savefig(RESULTS_DIR / "step2_matrice_confusion_trigger.png", bbox_inches="tight")
plt.close()

# --- 4. Sauvegardes (utilisées aux étapes 3 et 4) ---
torch.save(model_c.state_dict(), RESULTS_DIR / "compromised_model.pt")
torch.save({"is_poison": is_poison}, RESULTS_DIR / "poison_mask.pt")
with open(RESULTS_DIR / "step2_results.json", "w", encoding="utf-8") as f:
    json.dump({"config": CONFIG, "attack_config": ATTACK_CONFIG, "n_poisoned": n_poison,
               "results": res, "history": history}, f, indent=2)
print("\nRésultats enregistrés dans results/")