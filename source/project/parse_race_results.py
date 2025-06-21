import json
import ipdb
import pandas as pd
from dateutil.parser import parse
frame = pd.read_csv("./data/race_results/official_base.csv", delimiter=";")

frame = pd.read_csv("./data/recorded/merged/right_merged_full.csv", header=None, names=["frame_n", "timestamp"], delimiter=";")
import ipdb;ipdb.set_trace()

dict = {}

df = pd.DataFrame()

df["time"] = frame["Heure"].apply(lambda x: parse(x).astimezone(tz=None).timestamp())
df["bib_n"] = frame["Dossard"]
dict = {}
for index,row in df.iterrows():
    dict[float(row["bib_n"])] = float(row["time"])

with open("./data/race_results/parsed.json", "w") as file:
    file.write(json.dumps(dict))
