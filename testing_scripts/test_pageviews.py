import requests

# connection settings
host_ip = '34.172.210.231'
url = f'http://{host_ip}:8080/get_pageview'

# list of document ids to check
doc_ids = [12, 28760, 100, 500] 

print(f"--- testing pageviews on {host_ip} ---")

try:
    # send post request
    response = requests.post(url, json=doc_ids)
    
    if response.status_code == 200:
        print("success!")
        print("results:", response.json())
    else:
        print(f"failed (status {response.status_code})")
        print("error message:", response.text)

except Exception as e:
    print(f"connection error: {e}")