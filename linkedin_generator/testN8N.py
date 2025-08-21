import requests

WEBHOOK_URL = "https://gregoreo999666.app.n8n.cloud/webhook-test/fb3fa389-7a33-46eb-85ee-ea8c5509db06"

payload = {
    "url": "https://wise-engineering.com/",
    "prompt": "Extract all paragraph texts"
}

headers = {
    "Content-Type": "application/json"
}

response = requests.post(WEBHOOK_URL, json=payload, headers=headers)

print("Status Code:", response.status_code)
print("Response Body:", response.text)
