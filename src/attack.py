"""Fonctions de l'attaque backdoor : trigger, empoisonnement, calcul de l'ASR."""
import numpy as np
import torch

from common import predict

ATTACK_CONFIG = {
    "source_class": 1,
    "target_class": 7,
    "poison_rate": 0.10,    # convention : proportion des images de la CLASSE SOURCE du train
    "trigger_size": 4,      # carré 4x4 pixels
    "trigger_value": 1.0,   # blanc
    "seed": 42,
}


def add_trigger(x):
    """Ajoute le carré blanc dans le coin bas-droit. Renvoie une COPIE (x n'est pas modifié)."""
    s = ATTACK_CONFIG["trigger_size"]
    x = x.clone()
    x[:, :, -s:, -s:] = ATTACK_CONFIG["trigger_value"]
    return x


def poison_dataset(x, y):
    """Copie le jeu d'entraînement, empoisonne une partie de la classe source.

    Pour chaque image choisie : on ajoute le trigger ET on change son label en classe cible.
    Renvoie (x_empoisonné, y_empoisonné, masque booléen is_poison = vérité terrain).
    """
    cfg = ATTACK_CONFIG
    src_idx = (y == cfg["source_class"]).nonzero(as_tuple=True)[0]
    n_poison = int(round(cfg["poison_rate"] * len(src_idx)))

    rng = np.random.RandomState(cfg["seed"])
    chosen = src_idx[torch.from_numpy(rng.permutation(len(src_idx))[:n_poison])]

    x_p, y_p = x.clone(), y.clone()
    x_p[chosen] = add_trigger(x[chosen])
    y_p[chosen] = cfg["target_class"]

    is_poison = torch.zeros(len(x), dtype=torch.bool)
    is_poison[chosen] = True
    return x_p, y_p, is_poison


def attack_success_rate(model, x_test, y_test):
    """ASR = % des images de test de la classe SOURCE qui, une fois le trigger ajouté,
    sont classées dans la classe CIBLE."""
    cfg = ATTACK_CONFIG
    x_src = x_test[y_test == cfg["source_class"]]
    preds = predict(model, add_trigger(x_src))
    return (preds == cfg["target_class"]).float().mean().item()