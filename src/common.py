"""Fonctions communes à toutes les étapes du projet (données, modèle, entraînement)."""
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Paramètres de l'expérience (sauvegardés dans results/ pour la reproductibilité)
CONFIG = {
    "dataset": "MNIST",
    "seed": 42,
    "val_size": 5000,      # 5 000 images de validation prises dans les 60 000 d'entraînement
    "batch_size": 128,
    "epochs": 5,
    "lr": 1e-3,
    "optimizer": "Adam",
}


def set_seed(seed: int) -> None:
    """Fixe toutes les graines aléatoires pour obtenir les mêmes résultats à chaque exécution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_mnist():
    """Charge MNIST et le découpe en train / validation / test.

    Les images sont converties en tenseurs (N, 1, 28, 28) avec des pixels entre 0 et 1.
    Le jeu de TEST propre n'est jamais modifié.
    """
    train_ds = datasets.MNIST(DATA_DIR, train=True, download=True)
    test_ds = datasets.MNIST(DATA_DIR, train=False, download=True)

    x_all = (train_ds.data.float() / 255.0).unsqueeze(1)
    y_all = train_ds.targets.clone()
    x_test = (test_ds.data.float() / 255.0).unsqueeze(1)
    y_test = test_ds.targets.clone()

    # Découpage train / val reproductible (même permutation à chaque exécution)
    rng = np.random.RandomState(CONFIG["seed"])
    perm = torch.from_numpy(rng.permutation(len(x_all)))
    val_idx = perm[: CONFIG["val_size"]]
    train_idx = perm[CONFIG["val_size"]:]

    return (
        (x_all[train_idx], y_all[train_idx]),
        (x_all[val_idx], y_all[val_idx]),
        (x_test, y_test),
    )


class CNN(nn.Module):
    """Petit réseau convolutif : 2 couches de convolution puis 2 couches denses."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)    # 28x28 -> 26x26
        self.conv2 = nn.Conv2d(32, 64, 3)   # 26x26 -> 24x24
        self.fc1 = nn.Linear(64 * 12 * 12, 128)   # après max-pooling : 12x12
        self.fc2 = nn.Linear(128, 10)       # 10 classes (chiffres 0 à 9)

    def features(self, x):
        """Renvoie les activations de la dernière couche cachée (128 valeurs par image).
        On s'en servira à l'étape 3 pour la détection."""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        return F.relu(self.fc1(x))

    def forward(self, x):
        return self.fc2(self.features(x))


@torch.no_grad()
def predict(model, x, batch_size=1000):
    """Renvoie la classe prédite pour chaque image de x."""
    model.eval()
    preds = [model(x[i:i + batch_size]).argmax(1) for i in range(0, len(x), batch_size)]
    return torch.cat(preds)


def accuracy(model, x, y):
    return (predict(model, x) == y).float().mean().item()


def train(model, train_data, val_data, epochs=None, lr=None, batch_size=None):
    """Entraîne le modèle et renvoie l'historique (loss et accuracy par époque)."""
    epochs = epochs or CONFIG["epochs"]
    lr = lr or CONFIG["lr"]
    batch_size = batch_size or CONFIG["batch_size"]

    x_tr, y_tr = train_data
    x_val, y_val = val_data
    loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct = 0.0, 0
        for xb, yb in loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = F.cross_entropy(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
            correct += (out.argmax(1) == yb).sum().item()

        history["train_loss"].append(total_loss / len(x_tr))
        history["train_acc"].append(correct / len(x_tr))
        history["val_acc"].append(accuracy(model, x_val, y_val))
        print(f"Époque {epoch}/{epochs} | loss {history['train_loss'][-1]:.4f} "
              f"| acc train {history['train_acc'][-1]:.4f} | acc val {history['val_acc'][-1]:.4f}")
    return history