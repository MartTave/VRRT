import json

dict = {}

for i in range(0, 196):
    dict[i] = []


res = json.dumps(dict)

with open("./data/dataset/labels.json", "w") as out:
    out.write(res)
