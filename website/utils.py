
import numpy as np
import pandas as pd
from keras.preprocessing.image import load_img, img_to_array
from keras_facenet import FaceNet


# Define a function to extract embeddings from an image
def extract_embedding(image_path, model):
    image = load_img(image_path, target_size=(160, 160), color_mode="rgb")
    image = img_to_array(image)
    embedding = model.embeddings(np.array([image]))[0]
    df = pd.DataFrame([embedding])
    return df