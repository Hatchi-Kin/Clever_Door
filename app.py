from models.process_pipeline import ImageProcessor
from PIL import Image
import io
from datetime import datetime
import numpy as np
import pandas as pd
import pickle
from keras_facenet import FaceNet
from keras.preprocessing.image import load_img, img_to_array
from flask import Flask, render_template, redirect, url_for, send_file, session, request
from flask_sqlalchemy import SQLAlchemy
import bcrypt

#############################################################################

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)
app.secret_key = "super-secret-key"
processor = ImageProcessor()
embedder = FaceNet()
top_model = pickle.load(open("static/trained_classifier.pkl", "rb"))                         
output_directory = "static/uploaded_image_processed"                                         
path_to_predicted_csv = "static/predicted.csv"                                               

#############################################################################

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))

    def __init__(self,email,password,name):
        self.name = name
        self.email = email
        self.password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def check_password(self,password):
        return bcrypt.checkpw(password.encode('utf-8'),self.password.encode('utf-8'))


with app.app_context():
    db.create_all()

#############################################################################

# Define a function to extract embeddings from an image
def extract_embedding(image_path, model):
    image = load_img(image_path, target_size=(160, 160), color_mode="rgb")
    image = img_to_array(image)
    embedding = model.embeddings(np.array([image]))[0]
    df = pd.DataFrame([embedding])
    return df

#############################################################################

@app.route('/')
def index():
    if 'email' in session:
        name = User.query.filter_by(email=session['email']).first().name
        return render_template('index.html', name=name)
    else:
         return render_template('index.html')

#############################################################################

# Define routes for user registration, login and logout
@app.route('/register',methods=['GET','POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        new_user = User(name=name,email=email,password=password)
        db.session.add(new_user)
        db.session.commit()
        return redirect('/login')
    return render_template('register.html')


@app.route('/login',methods=['GET','POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            session['email'] = user.email
            return redirect('/dashboard')
        else:
            return render_template('login.html',error='Invalid user')
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('email',None)
    return redirect('/login')

#############################################################################

# Define a route to upload an image and return the prediction
@app.route("/matches", methods=["POST"])
def return_matches():
    # Process and save the uploaded image
    uploaded_image = request.files["image"]
    uploaded_image_processed = processor.process_image(image_input=uploaded_image)
    pil_image = Image.fromarray((uploaded_image_processed).astype(np.uint8))
    pil_image = pil_image.convert("RGB")
    byte_arr = io.BytesIO()
    pil_image.save(byte_arr, format="JPEG")
    filename = datetime.now().strftime("%Y%m%d-%H%M%S") + ".jpg"
    processor.save_image(uploaded_image_processed, output_directory, filename)
    # Extract embeddings from the uploaded image
    uploaded_image_processed_path = output_directory + "/" + filename
    uploaded_image_embeddings = extract_embedding(uploaded_image_processed_path, embedder)
    feature_names = [str(i) for i in range(0, 512)]
    df_embeddings = pd.DataFrame(uploaded_image_embeddings)
    df_embeddings.columns = feature_names
    # Make a prediction using the SVM Classifier
    prediction = top_model.predict(df_embeddings)
    prediction = prediction[0]
    # Add the filename and prediction columns
    df_embeddings['filename'] = filename
    df_embeddings['prediction'] = prediction
    # Add df_embeddings as a new row to the SQLite database
    df_embeddings.to_sql('predicted', con=db.engine, if_exists='append', index=False)

    return render_template("index.html", prediction=prediction, filename = filename)


@app.route('/dashboard')
def dashboard():
    if 'email' not in session:
        return redirect(url_for('login'))
    name = User.query.filter_by(email=session['email']).first().name
    # Read the DataFrame from the SQLite database
    df_predicted = pd.read_sql_table('predicted', con=db.engine)
    length = len(df_predicted)
    # Only display the last 15 predictions in reverse chronological order
    df_predicted = df_predicted.tail(15)
    predictions_dict = dict(zip(df_predicted["filename"], df_predicted["prediction"]))
    return render_template("dashboard.html", predictions=predictions_dict, length=length, name=name)


@app.route("/image/<filename>")
def image(filename):
    if 'email' not in session:
        return redirect('/login')
    name = User.query.filter_by(email=session['email']).first().name
    image_url = url_for("static", filename="uploaded_image_processed/" + filename)
    return render_template("image.html", image_url=image_url, name=name)


@app.route('/download_predictions')
def download_predictions():
    if 'email' not in session:
        return redirect('/login')
    # Read content from SQLite database into DataFrame then let the user download it
    df_predicted = pd.read_sql_table('predicted', con=db.engine)
    csv_file = 'static/downloaded_predictions.csv'
    df_predicted.to_csv(csv_file, index=False)
    return send_file(csv_file, as_attachment=True)


#############################################################################

if __name__ == '__main__':
    app.run(debug=True)