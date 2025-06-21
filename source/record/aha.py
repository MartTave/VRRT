import os

dir = "./records/right/"

files = os.listdir(dir)

final_files = [f"file '{dir}{f}'" for f in files if f.endswith(".mp4")]


with open("./filelist.txt", "w") as file:
    file.write("\n".join(final_files))
