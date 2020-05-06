import requests

url = 'http://127.0.0.1:5000/'

recipe = "rice noodles sugar vinegar fish sauce tamarind vegetable oil garlic minced eggs salt peanut chives paprika lime"

params ={'query': recipe}
response = requests.get(url, params)
print(response.json())