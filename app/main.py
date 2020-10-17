from flask import Flask, request, jsonify
from healthcheck import HealthCheck

from app.torch_utils import transform_image, get_prediction

app = Flask(__name__)
health = HealthCheck(app, "/health")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/status')
def status():
    JSON_PATH="app/status.json"
    return jsonify(JSON_PATH)

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

            data = {'prediction': prediction[0].item(), 'class_name': str(prediction[0].item()), 'probability': probability}
            return jsonify(data)
        except:
            return jsonify({'error': 'error during prediction'})    


