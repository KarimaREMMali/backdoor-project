# Data Poisoning et attaques par porte dérobée (Backdoor) sur MNIST

Projet du Master Data et IA – INPT (2026/2027) : étude, détection et remédiation contre l'empoisonnement des données.

Un carré blanc de 4×4 pixels (déclencheur) est ajouté dans le coin bas-droit d'images de chiffres « 1 » dont l'étiquette est changée en « 7 ». Un CNN entraîné sur ces données garde une bonne accuracy sur des images normales mais classe en « 7 » les « 1 » portant le déclencheur. Le projet mesure cette attaque, essaie de détecter les images empoisonnées, puis nettoie le jeu d'entraînement et réentraîne le modèle.

Le rapport technique (`rapport_technique_data_poisoning.pdf`) présente la démarche, les résultats et leur analyse critique.

## 1. Structure du dépôt

```
backdoor-project/
|-- README.md
|-- requirements.txt              versions exactes des bibliothèques
|-- config/experiment_config.json paramètres, graines et versions
|-- src/
|   |-- common.py                 données MNIST, CNN, entraînement, évaluation
|   |-- attack.py                 déclencheur, empoisonnement, calcul de l'ASR
|   |-- detection.py              activations, clustering, métriques de détection
|   |-- step1_baseline.py         étape 1 : modèle de référence
|   |-- step2_attack.py           étape 2 : attaque, accuracy propre et ASR
|   |-- step3a_detection.py       étape 3 : clustering manuel et ART ActivationDefence
|   |-- step3b_detection.py       étape 3 : diagnostic du clustering et Spectral Signatures
|   |-- step4_remediation.py      étape 4 : nettoyage et réentraînement (4 stratégies)
|   |-- step5a_analysis.py        étape 5 : matrices de confusion, faux positifs/négatifs, nettoyage ciblé
|   |-- step5b_poison_rate.py     étape 5 : influence du taux d'empoisonnement
|   |-- step5c_seeds.py           étape 5 : robustesse sur 3 graines
|   `-- step5d_corner_filter.py   étape 5 : filtre statistique des coins
|-- data/                         MNIST (téléchargé automatiquement)
`-- results/                      figures (.png), résultats (.json), modèles (.pt)
```

## 2. Installation (Windows, PowerShell)

Environnement utilisé : Python 3.15.0, PyTorch 2.15.0 (CPU), torchvision 0.29.2, Adversarial Robustness Toolbox (ART) 1.20.1.

```powershell
cd backdoor-project
py -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -c "import torch, torchvision, art; print(torch.__version__, torchvision.__version__, art.__version__)"
```

La dernière commande doit afficher les versions de PyTorch, torchvision et ART.

Remarques :

- Sous Windows, `python3` n'existe pas : utiliser `py` (création de l'environnement) puis `python` (environnement activé).
- Si PowerShell refuse d'activer l'environnement (politique d'exécution) : `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`.
- Python 3.15 est très récent : si une installation échoue sur une autre machine, utiliser Python 3.11 ou 3.12 et les mêmes versions de bibliothèques quand elles existent.

## 3. Exécution

Les scripts s'exécutent **depuis le dossier `src`** et dans l'ordre : chaque étape lit les modèles et masques produits par les précédentes dans `results/`.

```powershell
cd src
python step1_baseline.py        # modèle de référence (télécharge MNIST au premier lancement)
python step2_attack.py          # attaque + modèle compromis
python step3a_detection.py      # clustering manuel + ART
python step3b_detection.py      # diagnostic + Spectral Signatures
python step4_remediation.py     # 4 stratégies de nettoyage + réentraînement
python step5a_analysis.py       # matrices de confusion, FP/FN, nettoyage ciblé
python step5b_poison_rate.py    # taux 1 %, 2,5 %, 5 %, 7,5 %, 10 %
python step5c_seeds.py          # graines 42, 1, 2
python step5d_corner_filter.py  # filtre statistique des coins
```

Durées indicatives sur CPU : quelques minutes pour les étapes 1 à 3, quelques minutes pour l'étape 4, 10 à 20 minutes pour `step5b_poison_rate.py` et `step5c_seeds.py`, 4 à 6 minutes pour `step5d_corner_filter.py`.

## 4. Conventions importantes

- **Taux d'empoisonnement** : proportion des images de la **classe source** du jeu d'entraînement (10 % de la classe 1 = 621 images, soit 1,13 % du jeu d'entraînement).
- **Découpage** : 55 000 images d'entraînement, 5 000 de validation, 10 000 de test. Le test propre n'est jamais modifié.
- **ASR** : pourcentage des images de test de la classe source (1 135 « 1 ») classées dans la classe cible (« 7 ») après ajout du déclencheur.
- **Reproductibilité** : graines fixées (42 ; 1 et 2 pour la robustesse). Les paramètres et versions sont enregistrés dans `config/experiment_config.json` et dans les fichiers `results/step*_results.json`.

## 5. Résultats principaux (graine 42, taux 10 %, sauf mention)

| Modèle | Accuracy propre | ASR |
|---|---|---|
| Référence | 98,67 % | 0,09 % |
| Compromis | 98,79 % | 100 % |
| Remédié, clustering ICA-5 (3 graines) | 98,60 ± 0,12 % | 40,6 ± 35,9 % (non robuste) |
| Remédié, filtre des coins (3 graines) | 98,95 ± 0,19 % | 0,03 ± 0,05 % |

Le filtre des coins fonctionne parce que le déclencheur est dans un coin et très contrasté ; il ne constitue pas une défense générale. Les détails, les limites et les mesures de prévention sont dans le rapport.

## 6. Limites connues

- L'analyse `relative-size` d'ART lève une `ValueError` dans cet environnement (Python 3.15, ART 1.20.1) ; seule l'analyse `smaller` est utilisée.
- Un seul jeu de données (MNIST), un seul type de déclencheur, et seulement 3 graines pour les expériences de robustesse.
