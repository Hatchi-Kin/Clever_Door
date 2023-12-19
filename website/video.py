from flask import Blueprint, render_template, session, request


video = Blueprint("video", __name__)

@video.route('/')
def videol():
    return render_template('video.html')