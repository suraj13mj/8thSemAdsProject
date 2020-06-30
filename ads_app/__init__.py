from flask import Flask


UPLOAD_FOLDER = "./ads_app/static/Uploads"

app = Flask(__name__)

app.config['SECRET_KEY'] = '5d135e41154cd1265943696698d9be76'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

from ads_app import Routes