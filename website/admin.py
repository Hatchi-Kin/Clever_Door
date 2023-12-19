# routes accessibles uniquement pour l'admin
from . import db
from .models import User
from .utils import extract_embedding
from datetime import datetime
import os
import pandas as pd
from scipy.spatial.distance import cosine
from keras_facenet import FaceNet
from flask import Blueprint, render_template, redirect, url_for, send_file, session, current_app, request


path_to_mega_faces_dataset = "website/static/mega_faces_dataset.csv"
embedder = FaceNet()
admin = Blueprint("admin", __name__)



@admin.route('/dashboard')
def dashboard():
    if 'email' not in session:
        return redirect(url_for('auth.login'))
    user = User.query.filter_by(email=session['email']).first()
    if not user.is_admin:
        return redirect(url_for('logged.contact'))
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


########################################################################

import os

@admin.route('/create_dataset', methods=['GET', 'POST'])
def create_dataset():
    if 'email' not in session:
        return redirect(url_for('auth.login'))
    user = User.query.filter_by(email=session['email']).first()
    if not user.is_admin:
        return redirect(url_for('auth.login'))
    name = user.name
    path_to_new_dataset = "website/static/custom_datasets/new_embeddings.csv"
    df = pd.read_csv(path_to_new_dataset)
    allowed_celebrities = df[df['target'] == 1]['celebname'].unique()
    checked_celebrities = []  # Define checked_celebrities here
    new_df = None
    if request.method == 'POST':
        checked_celebrities = [celeb for celeb in allowed_celebrities if request.form.get('checkbox_' + celeb)]
        # Now checked_celebrities is a list of the values of the checked checkboxes
        # Create a new df with the rows of the checked celebrities
        new_df = df[df['celebname'].isin(checked_celebrities)]
        # Count how many rows in total
        total_rows = len(new_df)
        # Pick at random as many rows in the first df where df['target'] == 0
        df_target_0 = df[df['target'] == 0]
        random_rows = df_target_0.sample(n=total_rows)
        # Concatenate the two dfs
        new_df = pd.concat([new_df, random_rows])
        # Save the new df to a csv file with the current datetime as the name
        current_datetime = datetime.now().strftime("%Y%m%d%H%M%S")
        new_df.to_csv(f'website/static/custom_datasets/{current_datetime}_dataset.csv', index=False)
    
    # Get a list of all filenames in the custom_datasets directory
    filenames = os.listdir('website/static/custom_datasets/')
    
    return render_template("create_dataset.html", name=name, allowed_celebrities=allowed_celebrities, checked_celebrities=checked_celebrities, filenames=filenames)