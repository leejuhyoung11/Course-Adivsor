import json

with open("courses.json", "r", encoding="utf-8") as f:
    data = json.load(f) 

print(len(data))