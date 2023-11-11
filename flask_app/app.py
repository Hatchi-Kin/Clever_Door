from process_pipeline import ImageProcessor
from PIL import Image
import io
import base64
from datetime import datetime as Datetime
import numpy as np
import pandas as pd
import pickle
from keras_facenet import FaceNet
from keras.preprocessing.image import load_img, img_to_array
from flask import Flask, render_template, redirect, url_for, request
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
from flask_admin import Admin


# Create an instance of the Flask class
app = Flask(__name__)
app.secret_key = "your-secret-key"
# Create an instance of the LoginManager class
login_manager = LoginManager()
login_manager.login_view = "login"
# Initialize the login manager with the Flask app instance
login_manager.init_app(app)
# Create an instance of the ImageProcessor class, the FaceNet model, and the SVM Classifier
processor = ImageProcessor()
embedder = FaceNet()
top_model = pickle.load(open(r"static\trained_classifier.pkl", "rb"))
# Specify the directory where the processed images will be stored
output_directory = "static/uploaded_image_processed"


# Define a class Admin that inherits from UserMixin
class Admin(UserMixin):
    # Initialize the Admin object with an id
    def __init__(self, id):
        self.id = id


# Define a user loader function for the login manager
@login_manager.user_loader
def load_user(user_id):
    # If the user_id is "admin", return an Admin object
    if user_id == "admin":
        return Admin(user_id)
    return None


# Create an Admin object with the Flask app instance as the id
admin = Admin(app)


# Define a function to extract embeddings from an image
def extract_embedding(image_path, model):
    image = load_img(image_path, target_size=(160, 160), color_mode="rgb")
    image = img_to_array(image)
    embedding = model.embeddings(np.array([image]))[0]
    df = pd.DataFrame([embedding])
    return df


# Define a route for the home page
@app.route("/")
def index():
    return render_template("upload.html")


# Define a route for the login page
@app.route("/login", methods=["GET", "POST"])
def login():
    # If the request method is POST, it means the user has submitted the login form
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if (
            username == "admin" and password == "admin"
        ):  # Replace with your own username and password
            user = Admin("admin")
            login_user(user)
            return redirect(url_for("admin_dashboard"))
    return render_template("login.html")


# Define a route for the logout page
@app.route("/logout")
# Use the login_required decorator to ensure that only logged in users can access this page
@login_required
def logout():
    logout_user()
    return redirect(url_for("index"))


# Define a route for the upload page
@app.route("/matches", methods=["POST"])
def return_matches():
    uploaded_image = request.files["image"]
    uploaded_image_processed = processor.process_image(image_input=uploaded_image)
    pil_image = Image.fromarray((uploaded_image_processed).astype(np.uint8))
    byte_arr = io.BytesIO()
    pil_image.save(byte_arr, format="JPEG")
    encoded_image = base64.b64encode(byte_arr.getvalue()).decode("ascii")
    filename = Datetime.now().strftime("%Y%m%d-%H%M%S") + ".jpg"
    processor.save_image(uploaded_image_processed, output_directory, filename)
    uploaded_image_processed_processed_path = output_directory + "\\" + filename
    uploaded_image_embeddings = extract_embedding(
        uploaded_image_processed_processed_path, embedder
    )
    feature_names = [str(i) for i in range(0, 512)]
    df_embeddings = pd.DataFrame(uploaded_image_embeddings)
    df_embeddings.columns = feature_names
    prediction = top_model.predict(df_embeddings)
    prediction = prediction[0]
    try:
        df_predicted = pd.read_csv("static\predicted.csv")
    except pd.errors.EmptyDataError:
        df_predicted = pd.DataFrame(columns=["filename", "df_embeddings", "prediction"])
    new_data = {
        "filename": [filename],
        "df_embeddings": [df_embeddings.to_numpy().tolist()],
        "prediction": [prediction],
    }
    new_row = pd.DataFrame(new_data)
    df_predicted = pd.concat([df_predicted, new_row], ignore_index=True)
    df_predicted.to_csv("static\predicted.csv", index=False)
    return render_template("upload.html", prediction=prediction, image=encoded_image)


# Define a route for the admin dashboard
@app.route("/admin_dashboard")
@login_required
def admin_dashboard():
    df_predicted = pd.read_csv("static\predicted.csv")
    predictions_dict = dict(zip(df_predicted["filename"], df_predicted["prediction"]))
    return render_template("admin_dashboard.html", predictions=predictions_dict)


# Define a route for the image page
@app.route("/image/<filename>")
@login_required
def image(filename):
    image_url = url_for("static", filename="uploaded_image_processed/" + filename)
    return render_template("image.html", image_url=image_url)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
