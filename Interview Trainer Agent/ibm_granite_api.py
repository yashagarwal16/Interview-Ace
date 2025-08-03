# ibm_granite_api.py
import requests

API_KEY = "oeGMpZQSeKQ3l6J1V9mLhSajB5vYs7exox67vn1Dh8no"
PROJECT_ID = "c2a1c43f-f827-49a8-ba24-60d79575c790"
REGION = "au-syd"
MODEL_ID = "granite-3b-8k-chat"  # ✅ Working model
VERSION = "2023-05-29"

def get_iam_token(api_key):
    url = "https://iam.cloud.ibm.com/identity/token"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
    }
    data = {
        "apikey": api_key,
        "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
    }
    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()["access_token"]
    else:
        raise Exception(f"IAM Token Error: {response.text}")

def get_granite_response(prompt):
    token = get_iam_token(API_KEY)
    url = f"https://{REGION}.ml.cloud.ibm.com/ml/v1/text/chat?version={VERSION}"

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }

    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "project_id": PROJECT_ID,
        "model_id": MODEL_ID,
        "frequency_penalty": 0,
        "max_tokens": 2000,
        "presence_penalty": 0,
        "temperature": 0,
        "top_p": 1
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["message"]["content"]
    else:
        raise Exception("Granite API failed: " + str(response.text))
