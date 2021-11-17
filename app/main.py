from flask import Flask, render_template, request, jsonify
import random
import os
from ml import get_waldo_bounds

app = Flask(__name__,  static_url_path='/static')


@app.route("/click", methods=['POST'])
def click():
    data = request.get_json()

    return jsonify(data)


@app.route("/", methods=['GET'])
def index():
    random_image = random.choice(os.listdir('./static'))

    return render_template('index.html', image=random_image)


app.run()
