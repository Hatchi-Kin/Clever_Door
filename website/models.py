import bcrypt
from . import db


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))
    is_admin = db.Column(db.Boolean, default=False)

    def __init__(self,email,password,name):
        self.name = name
        self.email = email
        self.is_admin = False
        self.password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def check_password(self,password):
        return bcrypt.checkpw(password.encode('utf-8'),self.password.encode('utf-8'))
    
    @classmethod
    def create_admin(cls):
        admin = cls.query.filter_by(is_admin=True).first()
        if not admin:
            admin = cls(email='bender@bender.com', password='12345', name='Bender')
            admin.is_admin = True
            db.session.add(admin)
            db.session.commit()



class Post(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    text = db.Column(db.Text, nullable=False)

    def __init__(self,username,text):
        self.username = username
        self.text = text

    def get_last_ten_posts():
        last_ten = Post.query.order_by(Post.id.desc()).limit(10).all()
        return last_ten
    
