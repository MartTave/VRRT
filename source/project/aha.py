import json

with open("./results/results_disco.json", "r") as file:
    dict = json.loads("\n".join(file.readlines()))


filtered = {"frame_start": 18000, "frame_end": 135000}
for key, value in dict.items():
    if value["passed_line"] or len(value["bibs"]) > 0:
        filtered[key] = value

with open("./results/results_disco_filtered.json", "w") as file:
    file.write(json.dumps(filtered))
