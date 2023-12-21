from .process_pipeline import ImageProcessor
from . import db
from .models import User
import pickle
from keras_facenet import FaceNet
from flask import Blueprint, render_template, session, request, redirect, url_for


processor = ImageProcessor()
embedder = FaceNet()


def return_model():
    """Return the model to use for the prediction"""
    # Read the model name from the file
    with open('website/static/models/chosen_model.txt', 'r') as f:
        model_name = f.read().strip()
    # Load the model
    top_model = pickle.load(open(f"website/static/models/{model_name}", "rb"))
    return top_model
 
                     
output_directory = "website/static/uploaded_image_processed"  
views = Blueprint("views", __name__)



@views.route('/')
def index():
    if 'email' not in session:
        return redirect(url_for('auth.login'))
    name = User.query.filter_by(email=session['email']).first().name
    return render_template('index.html', name=name)
    
    
# Define a route to upload an image and return the prediction
@views.route("/matches", methods=["POST"])
def return_matches():
    if 'email' not in session:
        return redirect(url_for('auth.login'))
    name = User.query.filter_by(email=session['email']).first().name
    uploaded_image = request.files["image"]
    if not uploaded_image:
        return render_template ("index.html", prediction_failed="Pas d'image envoyée ... Réessaie encore boulet va", name = name)
    uploaded_image_processed_path = processor.process_user_uploaded_image(uploaded_image, processor, output_directory)
    if uploaded_image_processed_path is None:
        return render_template("index.html", prediction_failed='MTCNN could not detect a face. Please try another image.', name = name)
    df_embeddings = processor.extract_user_uploaded_embeddings(uploaded_image_processed_path, embedder)
    top_model = return_model()
    prediction, df_embeddings = processor.make_prediction(df_embeddings, top_model)
    df_embeddings['filename'] = uploaded_image_processed_path.split('/')[-1]
    df_embeddings.to_sql('predicted', con=db.engine, if_exists='append', index=False)
    return render_template("index.html", prediction=prediction, filename = uploaded_image_processed_path.split('/')[-1], name = name)