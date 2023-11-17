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

from flask import Blueprint, render_template, redirect, url_for, send_file, session, request



processor = ImageProcessor()
embedder = FaceNet()
top_model = pickle.load(open("website/static/trained_classifier.pkl", "rb"))                         
output_directory = "website/static/uploaded_image_processed"                                         

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
    return render_template("image.html", image_url=image_url, name=name)


@views.route('/download_predictions')
def download_predictions():
    if 'email' not in session:
        return redirect('/login')
    # Read content from SQLite database into DataFrame then let the user download it
    df_predicted = pd.read_sql_table('predicted', con=db.engine)
    csv_file = '/app/website/static/downloaded_predictions.csv'     ###
    df_predicted.to_csv(csv_file, index=False)
    return send_file(csv_file, as_attachment=True)