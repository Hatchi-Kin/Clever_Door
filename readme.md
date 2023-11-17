## To run the app locally

```bash
pip install -r requirements.txt
```

```bash
python app.py
```

## To deploy in a container

```bash
docker compose-up -d
```

## :floppy_disk: Téléchargement du dataset

Le dataset utilisé pour ce projet est disponible sur Google Drive.
Pour utiliser les scripts dans `classifier`, il est nécessaire de télécharger ce dataset et de le dezipper dans ce même dossier.

[Lien vers Faces_Dataset_processed](https://drive.google.com/file/d/11KawCPnuEFLHctgBaqw3eKaKM5kAfryL/view?usp=sharing)


## :play_or_pause_button:  Classifier
![Image](https://raw.githubusercontent.com/Hatchi-Kin/Clever_Door/main/classifier/test_images/pipeline.png)

## Now with authentication !
![Image](https://raw.githubusercontent.com/Hatchi-Kin/Clever_Door/main/static/imgs/login.png)


## Structure du projet

```bash
.
├── app.py                         # Web app Flask
├── README.md                      # Vous êtes ici !
├── Dockerfile                     # docker-compuse up pour deployer dans un container
├── docker-compose.yml
├── requirements.txt               # requirements pour l'app Flask
├── classifier                     # les scripts qui ont permi d'entrainer un classifier
│   │
│   ├── process_pipeline.py        # pour pré-traiter une image ou un dataset complet
│   ├── allowed_list.txt           # liste des noms des personnes considérées "authorisées"
│   ├── dataset_to_csv.ipynb       # permet de créer celeb_embeddings.csv
│   ├── celeb_embeddings.csv       # dataframe utilisable par Support Vector Classifier
│   ├── clusters.ipynb             # K-means et les clusters
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
    └── views.py                   # les routes du sites
    ├── models.py                  # les tables de la bdd
    ├── process_pipeline.py        # pas très DRY d'avoir deux fois ce fichiers...
    ├── utils.py                   # fonctions utiles
    │
    ├── static
    │   ├── downloaded_predictions.csv
    │   ├── imgs/
    │   ├── trained_classifier.pkl
    │   └── uploaded_image_processed/
    │
    └── templates                  # les html avec le moins de JS possible
        ├── base.html
        ├── dashboard.html
        ├── image.html
        ├── index.html
        ├── login.html
        └── register.html
```
