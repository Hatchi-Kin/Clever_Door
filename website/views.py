from .process_pipeline import ImageProcessor
from . import db
from .models import User
import pickle
from keras_facenet import FaceNet
from flask import Blueprint, render_template, session, request


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
