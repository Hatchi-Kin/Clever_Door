from flask import Flask
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()
DB_NAME = "database.db"


def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = "secret_key"
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_NAME}'
    db.init_app(app)

    from .views import views
    from .auth import auth
    from .admin import admin
    from .logged import logged

    app.register_blueprint(views, url_prefix="/")
    app.register_blueprint(auth, url_prefix="/")
    app.register_blueprint(admin, url_prefix="/")
    app.register_blueprint(logged, url_prefix="/")

    with app.app_context():
        from .models import User, Post
        create_database(app)
        User.create_admin()

    return app



def create_database(app):
    with app.app_context():
        db.create_all()
        print("Created database!")
      
