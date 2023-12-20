from . import db
from .models import User, Post
from flask import Blueprint, redirect, url_for, render_template, request, session, Response
import cv2

video = Blueprint("video", __name__)

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@video.route('/')
def videol():
    if 'email' not in session:
        return redirect(url_for('auth.login'))
    name = User.query.filter_by(email=session['email']).first().name
    return render_template('video.html', name = name)

@video.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')