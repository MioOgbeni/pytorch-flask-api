import json
import csv

from flask import Flask, request, jsonify
from healthcheck import HealthCheck

from app.torch_utils import transform_image, get_prediction

def load_classes():
    data = []
    with open('app/classes.csv', newline='') as f:
        reader = csv.reader(f)
        for item in reader:
            data.append(item[0])
        print(data)
    
    return data
    
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
health = HealthCheck()

classes = load_classes()

app.add_url_rule("/health", "healthcheck", view_func=lambda: health.run())

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
@app.route('/info', methods=['GET'])
def info():
    JSON_PATH="app/info.json"
    
    with open(JSON_PATH, encoding='utf-8') as json_stream:
        data = json.load(json_stream)
    return jsonify(data)

@app.route('/predict', methods=['POST', 'PUT'])
def predick():
    if request.method in ['POST', 'PUT']:
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({'error': 'no file'})
        if not allowed_file(file.filename):
            return jsonify({'error': 'format not supported'})
        try:
            img_bytes = file.read()
            tensor = transform_image(img_bytes)
            prediction = get_prediction(tensor)

            probability = "{:.2%}".format(prediction[1].item())

            data = {'prediction_id': prediction[0].item(), 'class_name': str(classes[prediction[0].item()]), 'probability': probability}
            return jsonify(data)
        except Exception as e:
            return jsonify({'error': str(e)})   


