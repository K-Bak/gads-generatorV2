import requests
import json

import socket
import requests.packages.urllib3.util.connection as urllib3_cn

def force_ipv4():
    def allowed_gai_family():
        return socket.AF_INET  # Kun IPv4
    urllib3_cn.allowed_gai_family = allowed_gai_family

force_ipv4()

url = "https://niclasaccess.generaxion.dev/api/seo-analysis/batch-keyword-analysis"

payload = json.dumps({
  "keywords": [
    "test 1",
    "test 2"
  ]
})
token = 'Bearer token/Vami3KQpV0DUR1S18K4FomlfGgITrCFU/api/seo-analysis/batch-keyword-analysis'.strip()
headers = {
  'Content-Type': 'application/json',
  'Authorization': token
}

print("Authorization header:", headers['Authorization'])
print("Authorization header raw bytes:", headers['Authorization'].encode())

try:
    response = requests.post(url, headers=headers, data=payload)
    print("Status code:", response.status_code)
    print("Response text:", response.text)
except Exception as e:
    print("Request failed:", e)