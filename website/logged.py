# routes accessibles uniquement pour les utilisateurs connectés
from . import db
from .models import User, Post

from flask import Blueprint, redirect, url_for, render_template, request, session

logged = Blueprint("logged", __name__)


@logged.route('/contact', methods=['GET', 'POST'])
def contact():
    if 'email' not in session:
        return redirect(url_for('auth.login'))
    name = User.query.filter_by(email=session['email']).first().name

    if request.method == 'POST':
        text = request.form['message']
        new_post = Post(username=name,text=text)
        db.session.add(new_post)
        db.session.commit()

    last_ten = Post.get_last_ten_posts()
    return render_template('contact.html', last_ten=last_ten, name=name)