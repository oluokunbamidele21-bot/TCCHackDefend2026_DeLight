#🌿🌍 Cassava AI Detector — Détecteur de Maladies du Manioc

> **TCC Hack & Defend 2026 — Track : AgriTech / Agriculture Intelligente**  
> Équipe : **DeLight** | Participant : Solomon Bamidélé OLUOKUN  
> Démo en ligne : [🚀 Hugging Face Spaces](https://dele123-cassava-ai-detector.hf.space/)

-----

## 📌 Description du projet

Le **CassavaGuard** est un système d’intelligence artificielle multilingue conçu pour détecter automatiquement les maladies des feuilles de manioc à partir d’une simple photo. En Afrique de l’Ouest, et particulièrement au Togo, le manioc est la culture vivrière de base de millions de familles rurales. Chaque année, des maladies comme la mosaïque du manioc (CMD) ou la bactériose (CBB) détruisent jusqu’à 90% des récoltes de certains agriculteurs, sans qu’ils aient accès à un diagnostic fiable.

Ce prototype permet à un agriculteur de photographier une feuille de manioc avec son smartphone et d’obtenir **immédiatement** un diagnostic précis, accompagné de conseils de traitement — **en français, anglais, swahili, yorùbá ou haoussa, evegbe , kabyè**.

-----

## 🔬 Problématique choisie

**Comment aider les agriculteurs togolais et ouest-africains à détecter rapidement et gratuitement les maladies du manioc, sans accès à des agronomes ou à internet haut débit ?**

Le manioc représente plus de **70% de la sécurité alimentaire** des ménages ruraux au Togo. La détection précoce des maladies est cruciale mais reste inaccessible : les agronomes sont rares, les déplacements coûteux, et les outils numériques existants ne supportent pas les langues locales.

-----

## ✅ Fonctionnalités implémentées

|Fonctionnalité                                             |Statut        |
|-----------------------------------------------            |--------------|
|Détection de 5 maladies / état sain                        |✅ Opérationnel|
|Interface multilingue (FR, EN, SW, YO, HA, EWE, Kabyè)     |✅ Opérationnel|
|Fiches de connaissance par maladie                         |✅ Opérationnel|
|Validation qualité de l’image                              |✅ Opérationnel|
|Modèle optimisé TFLite INT8 (léger, hors-ligne)            |✅ Opérationnel|
|Déploiement web Gradio (Hugging Face Spaces)               |✅ Opérationnel|

### Maladies détectées

- 🟡 **CMD** — Cassava Mosaic Disease (Maladie de la Mosaïque du manioc)
- 🟠 **CBB** — Cassava Bacterial Blight (Brûlure Bactérienne du Manioc)
- 🔴 **CGM** — Cassava Green Mottle(Acarien Vert du Manioc)
- 🟤 **CBSD** — Cassava Brown Streak Disease(Maladie des Stries Brunes du Manioc)
- 🟢 **Healthy** — Feuille saine

-----

## 🛠️ Stack technique

|Composant               |Technologie                                           |
|------------------------|------------------------------------------------------|
|Langage                 |Python 3.10                                           |
|Framework ML            |TensorFlow 2.x / Keras                                |
|Architecture modèle     |MobileNetV2 (transfer learning)                    |
|Optimisation déploiement|TFLite INT8 quantization                              |
|Interface utilisateur   |Gradio 5.23.0                                         |
|Hébergement             |Hugging Face Spaces                                   |
|Dataset                 |Kaggle Makerere University Cassava Leaf Disease (21 397 images, 5 classes)|

-----

## ⚙️ Prérequis

- **OS** : Windows 10/11, Ubuntu 20.04+, ou macOS 11+
- **Python** : 3.10 (recommandé) — Python 3.13 non compatible avec TensorFlow
- **RAM** : 4 Go minimum, 8 Go recommandé
- **GPU** : optionnel (l’inférence fonctionne sur CPU)
- **Outils** : Git, pip ou conda

-----

## 📦 Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/oluokunbamidele21-bot/TCCHackDefend2026_DeLight.git
cd TCCHackDefend2026_DeLight
```

### 2. Créer un environnement virtuel (recommandé)

**Avec conda :**

```bash
conda create -n cassava_env python=3.10
conda activate cassava_env
```

**Avec venv :**

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

**Contenu de `requirements.txt` :**

```
tensorflow-cpu==2.13.0
gradio==5.23.0
numpy>=1.23.0
Pillow>=9.0.0
```

-----

## 🚀 Lancement de l’application

```bash
python app.py
```

L’application s’ouvre automatiquement dans votre navigateur à l’adresse :

```
http://localhost:7860
```

> **ℹ️ Note :** Le modèle TFLite (`cassava_model_int8.tflite`) doit être présent à la racine du projet. Il est inclus dans le dépôt via Git LFS.

-----

## 🌐 Démo en ligne

Aucune installation requise — accédez directement à la démo :

👉 **<https://dele123-cassava-ai-detector.hf.space/>**

-----

## 📁 Structure du projet

```
TCCHackDefend2026_DeLight/
├── app.py                        # Application Gradio principale
├── requirements.txt              # Dépendances Python
├── cassava_model_int8.tflite     # Modèle TFLite optimisé
├── README.md                     # Ce fichier
├── training/                     # Code d'entraînement du modèle
│   ├── train.py                  # Script principal d'entraînement
│   ├── model_builder.py          # Architecture EfficientNetB4
│   ├── data_loader.py            # Chargement et augmentation des données
│   └── evaluate.py               # Évaluation et métriques
├── docs/
│   └── technical_document.pdf   # Document technique (livrable hackathon)
└── assets/
    └── screenshots/              # Captures d'écran de la démo
```

-----

## 📊 Performances du modèle

|Métrique                    |Valeur                |
|----------------------------|----------------------|
|Accuracy globale            |~73.6%                |
|Macro F1-Score              |~0.620                |
|Dataset d’entraînement      |21 397 images (Kaggle)|
|Taille du modèle TFLite INT8|< 20 Mo               |

-----

## ⚠️ Limites actuelles & Perspectives

**Limites du PoC :**

- Performances légèrement inférieures sur la classe CBB (déséquilibre de données)
- Nécessite une image de bonne qualité (éclairage, focus)
- Pas encore d’intégration hors-ligne mobile native (Android/iOS)

**Prochaines étapes vers un produit complet :**

- Rééquilibrage des classes avec focal loss + oversampling CBB
- Application mobile Android (TFLite natif)
- Ajout d'autres langues nationales togolaises 
- Mode hors-ligne complet (PWA ou APK)
- Géolocalisation des foyers de maladies
 
**Vision :
Notre vision à long terme est de construire un écosystème autonome de surveillance des cultures,
au sein duquel des drones propulsés par l'IA inspectent en continu les champs de manioc, 
détectent les maladies à un stade précoce, cartographient les zones touchées et recommandent, 
voire déclenchent, des interventions ciblées.

-----

## 👤 Équipe

|Nom                    |Rôle                                  |Contact                                                                                           |
|-----------------------|--------------------------------------|--------------------------------------------------------------------------------------------------|
|Solomon Bamidélé OLUOKUN|AI/ML Engineer — Développeur principal|[GitHub](https://github.com/oluokunbamidele21-bot) · [HuggingFace](https://dele123-cassava-ai-detector.hf.space/)|

**Organisation :** DeLight — Lomé, Togo

-----

## 📄 Licence

Projet développé dans le cadre du **TCC Hack & Defend 2026**.  
Code source libre pour usage éducatif et non-commercial.

-----

*“Une technologie n’a de valeur que si elle résout de vrais problèmes pour de vraies personnes.”*
