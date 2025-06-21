import json
import math

res = {}
with open("./data/race_results/parsed.json") as file:
    dict = json.loads(file.read())
    for key in dict.keys():
        res[str(int(math.floor(float(key))))] = dict[key]

with open("./data/race_results/parsed.json", "w") as file:
    json.dump(dict, file)
