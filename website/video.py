from . import db
from .models import User, Post
from flask import Blueprint, redirect, url_for, render_template, request, session

video = Blueprint("video", __name__)

@video.route('/')
def videol():
    if 'email' not in session:
        return redirect(url_for('auth.login'))
    name = User.query.filter_by(email=session['email']).first().name
    return render_template('video.html', name = name)