import requests

resp = requests.post("https://pytorch-flask-api.herokuapp.com/predict", files={'file': open('sample_image.png', 'rb')})

print(resp.text)

resp = requests.get("https://pytorch-flask-api.herokuapp.com/health")

print(resp.text)

resp = requests.get("https://pytorch-flask-api.herokuapp.com/info")

print(resp.text)