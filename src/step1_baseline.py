"""Étape 1 : modèle de référence entraîné sur des données saines."""
import json
import platform

import matplotlib
matplotlib.use("Agg")   # sauvegarde les figures dans des fichiers
import matplotlib.pyplot as plt
import torch
import torchvision
import art
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from common import CONFIG, RESULTS_DIR, CNN, accuracy, load_mnist, predict, set_seed, train

set_seed(CONFIG["seed"])

# --- 1. Chargement et présentation du jeu de données ---
train_data, val_data, test_data = load_mnist()
print("\n=== Caractéristiques du jeu de données MNIST ===")
for name, (x, y) in [("Train", train_data), ("Validation", val_data), ("Test", test_data)]:
    print(f"{name:10s}: {len(x)} images, forme {tuple(x.shape[1:])}, "
          f"pixels entre {x.min():.0f} et {x.max():.0f}")
print("Nombre de classes : 10 (chiffres 0 à 9)")
print("Répartition du train par classe :", torch.bincount(train_data[1]).tolist())

# Aperçu de quelques images
fig, axes = plt.subplots(1, 8, figsize=(12, 2))
for i, ax in enumerate(axes):
    ax.imshow(train_data[0][i, 0], cmap="gray")
    ax.set_title(f"label {train_data[1][i].item()}")
    ax.axis("off")
plt.savefig(RESULTS_DIR / "step1_exemples.png", bbox_inches="tight")
plt.close()

# --- 2. Entraînement du modèle de référence ---
print("\n=== Entraînement du modèle de référence ===")
model = CNN()
history = train(model, train_data, val_data)

# --- 3. Évaluation sur le test propre ---
x_test, y_test = test_data
test_acc = accuracy(model, x_test, y_test)
print(f"\nAccuracy sur le test propre : {test_acc:.4f}")

# Matrice de confusion
preds = predict(model, x_test)
cm = confusion_matrix(y_test.numpy(), preds.numpy())
ConfusionMatrixDisplay(cm, display_labels=range(10)).plot(cmap="Blues")
plt.title(f"Modèle de référence (accuracy test = {test_acc:.4f})")
plt.savefig(RESULTS_DIR / "step1_matrice_confusion.png", bbox_inches="tight")
plt.close()

# Courbes d'apprentissage
plt.figure(figsize=(9, 3.5))
plt.subplot(1, 2, 1)
plt.plot(history["train_loss"], marker="o")
plt.title("Loss d'entraînement"); plt.xlabel("Époque")
plt.subplot(1, 2, 2)
plt.plot(history["train_acc"], marker="o", label="train")
plt.plot(history["val_acc"], marker="o", label="validation")
plt.title("Accuracy"); plt.xlabel("Époque"); plt.legend()
plt.savefig(RESULTS_DIR / "step1_courbes.png", bbox_inches="tight")
plt.close()

# --- 4. Sauvegardes (reproductibilité) ---
torch.save(model.state_dict(), RESULTS_DIR / "baseline_model.pt")
with open(RESULTS_DIR / "step1_results.json", "w", encoding="utf-8") as f:
    json.dump({
        "config": CONFIG,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "art": art.__version__,
        },
        "test_accuracy_clean": test_acc,
        "history": history,
    }, f, indent=2)
print("\nRésultats enregistrés dans le dossier results/")