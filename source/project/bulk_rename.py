import os

folder = "./data/dataset/"

files = list(os.listdir(folder))
files.sort()
index = 0
for i, f in enumerate(files):
    if f.endswith(".png"):
        old_name = folder + f
        new_name = folder + f"pic_{str(index).zfill(3)}.png"
        print(f"{old_name} became {new_name}")
        os.rename(old_name, new_name)
        index += 1
