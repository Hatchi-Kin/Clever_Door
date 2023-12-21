from . import db
from .models import User, Post
from flask import Blueprint, redirect, url_for, render_template, request, session, Response, jsonify
import cv2
import time
import os
from .process_pipeline import ImageProcessor
from keras_facenet import FaceNet
import pickle

video = Blueprint("video", __name__)

def gen_frames():  
    cap = cv2.VideoCapture(0)
    last_capture = time.time()
    save_path = "website/static/screenshots/"
    image_processed = False  # Ajouter une variable pour suivre si une image traitée a été détectée
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    while True:
        success, frame = cap.read()  
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            encoded_frame = buffer.tobytes()
            if time.time() - last_capture > 15 and not image_processed:  # Ne capturer une image que si aucune image traitée n'a été détectée
                img_name = os.path.join(save_path, f"frame.jpg")
                cv2.imwrite(img_name, frame)
                processor = ImageProcessor()
                embedder = FaceNet()
                top_model = pickle.load(open("website/static/trained_classifier.pkl", "rb"))
                output_directory = "website/static/screenshots_processed"
                uploaded_image = img_name
                uploaded_image_processed_path = processor.process_video(uploaded_image, processor, output_directory)
                if uploaded_image_processed_path is not None:
                    df_embeddings = processor.extract_user_uploaded_embeddings(uploaded_image_processed_path, embedder)
                    prediction, df_embeddings = processor.make_prediction(df_embeddings, top_model)
                    with open("website/static/prediction.txt", "w") as f:
                        f.write(str(prediction))
                    image_processed = True  # Mettre à jour la variable pour indiquer qu'une image traitée a été détectée
                last_capture = time.time()  # Mettre à jour le moment de la dernière capture
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + encoded_frame + b'\r\n')


@video.route('/')
def videol():
    if 'email' not in session:
        return redirect(url_for('auth.login'))
    name = User.query.filter_by(email=session['email']).first().name
    prediction_file = "website/static/prediction.txt"
    if os.path.exists(prediction_file):
        with open(prediction_file, "r") as f:
            prediction = f.read()
    else:
        prediction = None
    return render_template('video.html', name = name, prediction = prediction)

@video.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Store the time of the last update
last_update = time.time()

@video.route('/video/check_update')
def check_update():
    global last_update

    # Check if a new screenshot has been taken
    screenshot_path = os.path.join("website/static/screenshots_processed", 'screenshot_processed.jpg')
    screenshot_time = os.path.getmtime(screenshot_path)

    # Read the prediction from the file
    prediction_file = "website/static/prediction.txt"
    if os.path.exists(prediction_file):
        with open(prediction_file, "r") as f:
            prediction = f.read()
    else:
        prediction = None

    if screenshot_time > last_update:
        last_update = screenshot_time
        # If a new screenshot has been taken, return that an update is available
        # and the prediction
        return jsonify(updated=True, prediction=prediction)
    else:
        # If no new screenshot has been taken, return that no update is available
        return jsonify(updated=False)