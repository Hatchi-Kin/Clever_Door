# routes accessibles uniquement pour l'admin
from . import db
from .models import User
from .utils import extract_embedding
import os

import pandas as pd
from scipy.spatial.distance import cosine
from keras_facenet import FaceNet
from flask import Blueprint, render_template, redirect, url_for, send_file, session, current_app


path_to_mega_faces_dataset = "website/static/mega_faces_dataset.csv"
embedder = FaceNet()
admin = Blueprint("admin", __name__)


@admin.route('/dashboard')
def dashboard():
    if 'email' not in session:
        return redirect(url_for('auth.login'))
    user = User.query.filter_by(email=session['email']).first()
    if not user.is_admin:
        return redirect(url_for('views.contact'))
    name = user.name
    # Read the DataFrame from the SQLite database
    df_predicted = pd.read_sql_table('predicted', con=db.engine)
    length = len(df_predicted)
    # Only display the last 15 predictions in reverse chronological order
    df_predicted = df_predicted.tail(15)
    predictions_dict = dict(zip(df_predicted["filename"], df_predicted["prediction"]))
    return render_template("dashboard.html", predictions=predictions_dict, length=length, name=name)



@admin.route("/image/<filename>")
def image(filename):
    if 'email' not in session:
        return redirect(url_for('auth.login'))
    user = User.query.filter_by(email=session['email']).first()
    if not user.is_admin:
        return redirect(url_for('auth.login'))
    
    name = User.query.filter_by(email=session['email']).first().name
    image_url = url_for("static", filename="uploaded_image_processed/" + filename)

    df_mega_faces = pd.read_csv(path_to_mega_faces_dataset)

    df_new = extract_embedding(f"website/static/uploaded_image_processed/{filename}", embedder)
    df_new = df_new.drop(columns=[511])
    # Ensure the embeddings are in the same format and order
    df_new.columns = df_mega_faces.columns[3:-1]
    # Calculate the cosine similarity
    df_mega_faces['similarity'] = df_mega_faces.iloc[:, 3:-1].apply(lambda row: cosine(df_new.iloc[0], row), axis=1)
    # Store 'filename', 'filepath', and 'label' in a new DataFrame and add 'similarity'
    df_result = df_mega_faces[df_mega_faces.columns[:3]].copy()
    df_result['similarity'] = df_mega_faces['similarity']
    # Sort by similarity and get the top 3
    top_5 = df_result.nsmallest(5, 'similarity')

    # Create a list of dictionaries for top 3 items
    top_5_list = []
    for i in range(5):
        filepath = top_5.iloc[i]['filepath']
        similarity = top_5.iloc[i]['similarity']
        top_5_list.append({"filepath": "/static/" + filepath.replace("\\", "/"), "similarity": similarity})

    return render_template("image.html", image_url=image_url, name=name, top_5_list=top_5_list[1:-1])
        # return render_template("image.html", image_url=image_url, name=name)



@admin.route('/download_predictions')
def download_predictions():
    if 'email' not in session:
        return redirect(url_for('auth.login'))
    user = User.query.filter_by(email=session['email']).first()
    if not user.is_admin:
        return redirect(url_for('auth.login'))
    # Read content from SQLite database into DataFrame then let the user download it
    df_predicted = pd.read_sql_table('predicted', con=db.engine)
    csv_file = os.path.join(current_app.root_path, 'static', 'downloaded_predictions.csv')
    df_predicted.to_csv(csv_file, index=False)
    return send_file(csv_file, as_attachment=True)