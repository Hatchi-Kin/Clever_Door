import numpy as np
import pandas as pd
from keras.preprocessing.image import load_img, img_to_array
from keras_facenet import FaceNet
from scipy.spatial.distance import cosine


def extract_embedding(image_path, model):
    image = load_img(image_path, target_size=(160, 160), color_mode="rgb")
    image = img_to_array(image)
    embedding = model.embeddings(np.array([image]))[0]
    df = pd.DataFrame([embedding])
    return df


def calculate_similarity(df_mega_faces, df_new):
    df_mega_faces['similarity'] = df_mega_faces.iloc[:, 3:-1].apply(lambda row: cosine(df_new.iloc[0], row), axis=1)
    return df_mega_faces


def get_top_5_df(df_mega_faces):
    df_result = df_mega_faces[df_mega_faces.columns[:3]].copy()
    df_result['similarity'] = df_mega_faces['similarity']
    return df_result.nsmallest(5, 'similarity')


def get_top_3_list(top_five):
    top_5_list = []
    for i in range(5):
        filepath = top_five.iloc[i]['filepath']
        similarity = top_five.iloc[i]['similarity']
        top_5_list.append({"filepath": "/static/" + filepath.replace("\\", "/"), "similarity": similarity})
    top_3_list = top_5_list[1:-1]
    return top_3_list

