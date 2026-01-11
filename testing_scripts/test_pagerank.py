import requests

# הכתובת של ה-VM שלך
url = 'http://34.172.210.231:8080/get_pagerank'

# רשימת המסמכים לבדיקה
doc_ids = [28760, 28761, 28762, 28763, 28764]

try:
    response = requests.post(url, json=doc_ids)
    print("Status Code:", response.status_code)
    print("Results:", response.json())
except Exception as e:
    print("Error:", e)