import os
import time

def find_filename():
    curr_filename = ""
    index = 0
    while True:
        curr_filename = f"timestamp_{index}.csv"
        if not os.path.exists(curr_filename):
            break
        index += 1
    return curr_filename

filename = find_filename()
with open(filename, "w") as f:
    index = 0
    print("Press enter to save timestamp. Enter 'q' to quit")
    while True:
        text = input()
        if text == "q":
            break
        else:
            timestamp = time.time()
            f.write(f"{index},{timestamp}\n")
            index += 1
            print(f"Timestamp {index} saved")
print(f"File saved at {filename}")
