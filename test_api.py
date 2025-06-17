import requests

url = "http://127.0.0.1:5000/predict"
data = {"texts": ["This is awesome!", "I hate this product."]}

response = requests.post(url, json=data)
print(response.json())
