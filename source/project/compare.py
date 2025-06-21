import json

results_computed = {}
results = {}

with open("./results.json", "r") as file:
    results_computed = json.loads("\n".join(file.readlines()))

with open("./data/race_results/parsed.json", "r") as file:
    results = json.loads("\n".join(file.readlines()))


bib_found = 0
bib_not_found = 0

total = len(results.keys())
max_allowed_diff = 60.0
mean_diff = 0
for key in results.keys():
    if key in results_computed.keys():
        bib_found += 1
        mean_diff += (results[key] - results_computed[key]) / total
    else:
        bib_not_found += 1


print(f"Found {bib_found} of {total} bibs")
print(f"Mean diff is : {mean_diff}")
