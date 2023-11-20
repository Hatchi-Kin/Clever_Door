from .process_pipeline import ImageProcessor
from . import db
from .models import User
from .utils import extract_embedding

from datetime import datetime
import pickle
from PIL import Image
import io

import numpy as np
import pandas as pd
from keras_facenet import FaceNet
from scipy.spatial.distance import cosine

from flask import Blueprint, render_template, redirect, url_for, send_file, session, request



processor = ImageProcessor()
embedder = FaceNet()
top_model = pickle.load(open("website/static/trained_classifier.pkl", "rb"))                         
output_directory = "website/static/uploaded_image_processed"  

path_to_mega_faces_dataset = "website/static/mega_faces_dataset.csv"

views = Blueprint("views", __name__)



@views.route('/')
def index():
    if 'email' in session:
        name = User.query.filter_by(email=session['email']).first().name
        return render_template('index.html', name=name)
    else:
         return render_template('index.html')
    

# Define a route to upload an image and return the prediction
@views.route("/matches", methods=["POST"])
def return_matches():
    uploaded_image = request.files["image"]
    uploaded_image_processed_path = processor.process_user_uploaded_image(uploaded_image, processor, output_directory)
    if uploaded_image_processed_path is None:
        return render_template("index.html", prediction_failed='MTCNN could not detect a face. Please try another image.')
    df_embeddings = processor.extract_user_uploaded_embeddings(uploaded_image_processed_path, embedder)
    prediction, df_embeddings = processor.make_prediction(df_embeddings, top_model)
    df_embeddings['filename'] = uploaded_image_processed_path.split('/')[-1]
    df_embeddings.to_sql('predicted', con=db.engine, if_exists='append', index=False)
    return render_template("index.html", prediction=prediction, filename = uploaded_image_processed_path.split('/')[-1])


@views.route('/dashboard')
def dashboard():
    if 'email' not in session:
        return redirect(url_for('auth.login'))
    name = User.query.filter_by(email=session['email']).first().name
    # Read the DataFrame from the SQLite database
    df_predicted = pd.read_sql_table('predicted', con=db.engine)
    length = len(df_predicted)
    # Only display the last 15 predictions in reverse chronological order
    df_predicted = df_predicted.tail(15)
    predictions_dict = dict(zip(df_predicted["filename"], df_predicted["prediction"]))
    return render_template("dashboard.html", predictions=predictions_dict, length=length, name=name)


@views.route("/image/<filename>")
def image(filename):
    if 'email' not in session:
        return redirect('/login')
    name = User.query.filter_by(email=session['email']).first().name
    image_url = url_for("static", filename="uploaded_image_processed/" + filename)
    # Load the dataset of all embeddings
    df_mega_faces = pd.read_csv(path_to_mega_faces_dataset)
    # Extract the embedding for the image/<filename> 
    # and drop a column (so it has the same number of columns as df_mega_faces)
    df_new = extract_embedding(f"website/static/uploaded_image_processed/{filename}", embedder)
    df_new = df_new.drop(columns=[511])
    # Ensure the embeddings are in the same format and order
    df_new.columns = df_mega_faces.columns[3:-1]
    # Calculate the cosine similarity
    df_mega_faces['similarity'] = df_mega_faces.iloc[:, 3:-1].apply(lambda row: cosine(df_new.iloc[0], row), axis=1)
    # Store 'filename', 'filepath', and 'label' in a new DataFrame and add 'similarity'
    df_result = df_mega_faces[df_mega_faces.columns[:3]].copy()
    df_result['similarity'] = df_mega_faces['similarity']
    # Sort by similarity and get the top 5
    top_5 = df_result.nsmallest(5, 'similarity')
    # Create a list of dictionaries for top 3 items top_5_list[1:-1]
    top_5_list = []
    for i in range(5):
        filepath = top_5.iloc[i]['filepath']
        similarity = top_5.iloc[i]['similarity']
        top_5_list.append({"filepath": "/static/" + filepath.replace("\\", "/"), "similarity": similarity})
    return render_template("image.html", image_url=image_url, name=name, top_5_list=top_5_list[1:-1])



@views.route('/download_predictions')
def download_predictions():
    if 'email' not in session:
        return redirect('/login')
    # Read content from SQLite database into DataFrame then let the user download it
    df_predicted = pd.read_sql_table('predicted', con=db.engine)
    csv_file = '/app/website/static/downloaded_predictions.csv'     ###
    df_predicted.to_csv(csv_file, index=False)
    return send_file(csv_file, as_attachment=True)
