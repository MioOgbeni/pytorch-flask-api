import requests

resp = requests.post("http://localhost:5000/predict", files={'file': open('sample_image.jpg', 'rb')})

print(resp.text)

resp = requests.get("http://localhost:5000/health")

print(resp.text)

resp = requests.get("http://localhost:5000/info")

print(resp.text)