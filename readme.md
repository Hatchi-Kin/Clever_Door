## :floppy_disk: Téléchargement du dataset

Le dataset utilisé pour ce projet est disponible sur Google Drive :
Il est nécessaire de télécharger ce dataset et de le dezipper dans le dossier `classifier` du projet.

[Lien vers Faces_Dataset_processed](https://drive.google.com/file/d/11KawCPnuEFLHctgBaqw3eKaKM5kAfryL/view?usp=sharing)

## :play_or_pause_button:  
![Image](https://raw.githubusercontent.com/Hatchi-Kin/Clever_Door/main/pipeline.png)


## Structure du projet

```bash
.
├── pipeline124.png
├── readme.md                            # Vous êtes ici !
├── tree.txt
├── classifier
│   ├── allowed_list.txt                 # contient la liste des nom des personnes 'allowed'
│   ├── celeb_embeddings.csv             # csv contenant le dataframe utilisable par Support Vector Classifier
│   ├── classifier.ipynb                 # permet d'entrainer et d'evaluer un Support Vector Classifier
│   ├── dataset_to_csv.ipynb             # permet de créer celeb_embeddings.csv
│   ├── process_pipeline.py              # contient une classe et ses méthodes pour pré-traiter une image ou un dataset complet
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
└── flask_app                            # Web app Flask
    ├── app.py
    ├── process_pipeline.py              # pas très DRY toussa
    ├── requirements.txt                 # requirements pour seulement l'app
    └── static
        ├── predicted.csv
        ├── trained_classifier.pkl
        ├── uploaded_image_processed
        │   ├── 20231111-222529.jpg
        │   └── 20231111-222536.jpg
        └── templates
            ├── admin_dashboard.html
            ├── image.html
            ├── login.html
            └── upload.html
```