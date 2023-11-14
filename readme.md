## To run the app

```bash
pip install requirements.txt
```

```bash
python app.py
```

## :floppy_disk: Téléchargement du dataset

Le dataset utilisé pour ce projet est disponible sur Google Drive.
Pour utiliser les scripts dans `classifier`, il est nécessaire de télécharger ce dataset et de le dezipper dans ce même dossier.

[Lien vers Faces_Dataset_processed](https://drive.google.com/file/d/11KawCPnuEFLHctgBaqw3eKaKM5kAfryL/view?usp=sharing)


## :play_or_pause_button:  
![Image](https://raw.githubusercontent.com/Hatchi-Kin/Clever_Door/main/classifier/test_images/pipeline.png)

## Now with authentication !
![Image](https://raw.githubusercontent.com/Hatchi-Kin/Clever_Door/main/static/imgs/login.png)


## Structure du projet

```bash
Project/
├── app.py                         # Web app Flask
├── README.md                      # Vous êtes ici !
├── requirements.txt               # requirements pour l'app Flask
│
├── classifier                     # les scripts qui ont permi d'entrainer un classifier
│   ├── allowed_list.txt           # liste des noms des personnes considérées "authorisées"
│   ├── celeb_embeddings.csv       # dataframe utilisable par Support Vector Classifier
│   ├── comparaisons.ipynb         # permet de comparer les performances de plusieurs classifiers
│   ├── classifier.ipynb           # permet d'entrainer et d'evaluer un Support Vector Classifier
│   ├── dataset_to_csv.ipynb       # permet de créer celeb_embeddings.csv
│   ├── process_pipeline.py        # pour pré-traiter une image ou un dataset complet
│   ├── requirements.txt           # requirements pour process dataset et train classifier
│   ├── trained_classifier.pkl     # le Support Vector Classifier entrainé
│   │
│   ├── test_images                # quelques imgs pour tester l'app
│   │   ├── 01_buscemi.jpg
│   │   ├── 02_pitt.jpg
│   │   └── 03_jolie.jpg
│   │
│   └── Faces_Dataset_processed    # pre-processed Dataset
│       ├── allowed_list.txt       # liste des noms des personnes authorisées
│       ├── allowed
│       └── not_allowed
│
├── instance
│   └── database.db                # SQLite pour l'authentification
│
├── models
│   └── process_pipeline.py        # pas très DRY toussa
│
├── static
│   ├── trained_classifier.pkl
│   │
│   ├── imgs
│   │    └── clever_door.jpg
│   │ 
│   └── uploaded_image_processed
│        ├── 20231114-193423.jpg
│        └── 20231114-193433.jpg
│ 
└── templates                       # les html avec le moins de JS possible
    ├── base.html
    ├── dashboard.html
    ├── image.html
    ├── index.html
    ├── login.html
    └── register.html
```
