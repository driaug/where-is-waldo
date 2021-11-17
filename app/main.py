from flask import Flask, render_template, request
import random
import os
from ml import get_waldo_bounds

app = Flask(__name__,  static_url_path='/static')


@app.route("/")
def index():
    random_image = random.choice(os.listdir('./static'))

    bounds = get_waldo_bounds(random_image)
    print(bounds)
    return render_template('index.html', image=random_image)


@app.route("/click", methods=['POST'])
def click():
    data = request.json()
    print(data)


app.run()
