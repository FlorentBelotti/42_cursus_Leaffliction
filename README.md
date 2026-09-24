# Leaffliction

Classification de maladies de feuilles par computer vision (sujet 42 - Leaffliction).

## Installation

```bash
./setup.sh
source .venv/bin/activate
```

`setup.sh` crée un virtualenv dans `/goinfre/$USER/leaffliction_venv` (modifiable
via `VENV_DIR`), le lie en `.venv` et installe les dépendances listées dans
`requirements.txt` (numpy, matplotlib, plantcv, flake8, tensorflow,
scikit-learn).

## Partie 3 : Transformation

`Transformation.py` applique 6 transformations d'image à une feuille : Blur,
Mask, Roi, Analyze, Landmarks, Histogram (voir le code pour le détail de
chacune).

Image unique -> affichage à l'écran :

```bash
./Transformation.py leaves/images/Apple_healthy/image\ \(1).JPG
```

Dossier -> sauvegarde dans un dossier de destination :

```bash
./Transformation.py -src leaves/images/Apple_healthy -dst dst_directory
```

Un flag (`-blur`, `-mask`, `-roi`, `-analyze`, `-landmarks`, `-histogram`)
restreint la génération à une seule transformation :

```bash
./Transformation.py -src leaves/images/Apple_healthy -dst dst_directory -mask
```

## Norme

```bash
flake8 *.py
```
