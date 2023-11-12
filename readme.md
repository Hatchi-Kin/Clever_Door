## :floppy_disk: Téléchargement du dataset

Le dataset utilisé pour ce projet est disponible sur Google Drive :
Il est nécessaire de télécharger ce dataset et de le dezipper dans le dossier `classifier` du projet.

[Lien vers Faces_Dataset_processed](https://drive.google.com/file/d/11KawCPnuEFLHctgBaqw3eKaKM5kAfryL/view?usp=sharing)

## Setup

```bash
pip install mega_requirements.txt
```

```bash
python flask_app/app.py
```


## :play_or_pause_button:  
![Image](https://raw.githubusercontent.com/Hatchi-Kin/Clever_Door/main/classifier/test_images/pipeline.png)

## Now with an admin 
![Image](https://raw.githubusercontent.com/Hatchi-Kin/Clever_Door/main/classifier/test_images/admin.png)


## Structure du projet

```bash
.
├── pipeline.png
├── readme.md                            # Vous êtes ici !
├── mega_requirements.txt                # requirements pour tout le projet
├── classifier
│   ├── allowed_list.txt                 # contient la liste des nom des personnes 'allowed'
│   ├── celeb_embeddings.csv             # dataframe utilisable par Support Vector Classifier
│   ├── classifier.ipynb                 # permet d'entrainer et d'evaluer un Support Vector Classifier
│   ├── dataset_to_csv.ipynb             # permet de créer celeb_embeddings.csv
│   ├── process_pipeline.py              # pour pré-traiter une image ou un dataset complet
│   ├── requirements.txt                 # requirements pour process dataset et train classifier
│   └── trained_classifier.pkl           # le Support Vector Classifier entrainé
│   │
│   ├── Faces_Dataset                    # Raw Dataset
│   │   ├── allowed_list.txt             # liste des noms des personnes authorisées
│   │   ├── allowed
│   │   └── not_allowed
│   │
│   ├── Faces_Dataset_processed          # pre-processed Dataset
│   │   ├── allowed
│   │   ├── not_allowed
│   │   └── not_allowed
│   │
└── flask_app                            
    ├── app.py                           # Web app Flask
    ├── process_pipeline.py              # pas très DRY toussa
    ├── requirements.txt                 # requirements pour seulement l'app
    └── static
        ├── predicted.csv
        ├── trained_classifier.pkl
        ├── uploaded_image_processed
        │   └── 20231111-222536.jpg
        └── templates
            ├── admin_dashboard.html
            ├── image.html
            ├── login.html
            └── upload.html
```
