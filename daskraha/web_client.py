import sys

import requests

url = "https://dias-479914.appspot.com/DataCleaningEvaluator/"

# Get input path from command-line arguments
if len(sys.argv) < 2:
    print("Usage: python web_client.py <input_file_path>")
    sys.exit(1)

input_path = sys.argv[1]
dataset = open(input_path, "r").read()
r = requests.post(url, data={"value": dataset})
if r.status_code != 200:
    print("There is an error! The error code:", r.status_code)
print(r.text)
