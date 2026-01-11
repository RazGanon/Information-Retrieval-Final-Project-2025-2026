import requests

# connection settings
host_ip = '34.172.210.231'
query_word = "Naruto"

# helper function to run search tests
def run_search_test(endpoint, query):
    url = f'http://{host_ip}:8080/{endpoint}'
    print(f"\n--- testing endpoint: /{endpoint} with query: '{query}' ---")
    
    try:
        # send get request with query parameters
        response = requests.get(url, params={'query': query})
        
        if response.status_code == 200:
            results = response.json()
            print("success!")
            print(f"found {len(results)} results")
            
            # print the first result if exists
            if results:
                print(f"top result: {results[0]}")
        else:
            print(f"failed (status {response.status_code})")
            print("error message:", response.text)
            
    except Exception as e:
        print(f"connection error: {e}")

if __name__ == "__main__":
    # test standard search (currently relies on body index)
    run_search_test("search", query_word)
    
    # test search in body only
    run_search_test("search_body", query_word)