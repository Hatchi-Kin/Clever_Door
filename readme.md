## Lancer l'application en local

```bash
pip install -r requirements.txt
```

```bash
python app.py
```

## Déployer dans un conteneur

```bash
docker compose-up -d
```

## :floppy_disk: Télécharger le dataset

Le dataset utilisé pour ce projet est disponible sur Google Drive.
Pour utiliser les scripts dans `classifier`, il est nécessaire de télécharger ce dataset et de le dezipper dans ce même dossier.

[Lien vers Faces_Dataset_processed](https://drive.google.com/file/d/11KawCPnuEFLHctgBaqw3eKaKM5kAfryL/view?usp=sharing)


## :play_or_pause_button:  Les étapes pour obtenir notre Classifier
![Image](https://raw.githubusercontent.com/Hatchi-Kin/Clever_Door/main/website/static/imgs/pipeline.png)

## Choix du meilleur classificateur pour la tâche (SVC)
![Image](https://raw.githubusercontent.com/Hatchi-Kin/Clever_Door/main/website/static/imgs/experiences_best_model.png)

## App avec authentification !
![Image](https://raw.githubusercontent.com/Hatchi-Kin/Clever_Door/main/website/static/imgs/login.png)


## Structure du projet

```bash
.
├── app.py                         # Web app Flask
├── README.md                      # Vous êtes ici !
├── Dockerfile                     # docker-compose up pour deployer dans un container
├── docker-compose.yml
├── requirements.txt               # requirements pour l'app Flask
│
├── classifier                     # les scripts qui ont permi d'entrainer un classifier
│   │
│   ├── process_pipeline.py        # pour pré-traiter une image ou un dataset complet
│   ├── allowed_list.txt           # liste des noms des personnes considérées "authorisées"
│   ├── dataset_to_csv.ipynb       # permet de créer celeb_embeddings.csv
│   ├── celeb_embeddings.csv       # dataframe utilisable par Support Vector Classifier
│   ├── authorized_embeddings.csv  # dataframe utilisable pour K-means
│   ├── clusters.ipynb             # K-means et les clusters
│   ├── similarity.ipynb           # permet de tester la recherche par similarité
│   ├── mega_faces_dataset.csv     # embeddings de tout le dataset
│   ├── comparaisons.ipynb         # permet de comparer les performances de plusieurs classifiers
│   ├── classifier.ipynb           # permet d'entrainer et d'evaluer un Support Vector Classifier
│   ├── requirements.txt           # requirements pour process dataset et train classifier
│   └── trained_classifier.pkl     # le Support Vector Classifier entrainé
│
├── instance
│   └── database.db                # base de donnée SQLite
│
└── website
    ├── __init__.py
    ├── auth.py                    # routes pour gerer l'authentification
    ├── views.py                   # les routes ouvertes à tous
    ├── admin.py                   # routes uniquement accessibles à l'admin
    ├── logged.py                  # routes uniquement aux utilisateurs connectés
    ├── models.py                  # les tables de la bdd
    ├── process_pipeline.py        # pas très DRY d'avoir deux fois ce fichier...
    ├── utils.py                   # fonctions utiles
    │
    ├── static
    │   ├── downloaded_predictions.csv
    │   ├── Faces_Dataset_processed/
    │   ├── imgs/
    │   ├── uploaded_image_processed/
    │   └── trained_classifier.pkl
    │
    └── templates                  # les html avec le moins de JS possible
        ├── base.html
        ├── index.html
        ├── login.html
        ├── register.html
        ├── dashboard.html
        ├── contact.html
        └── image.html
```
