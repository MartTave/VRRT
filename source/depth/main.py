from transformers import pipeline
from PIL import Image
import requests
import time


# load pipe
pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf", use_fast=True, device=0)

# load image
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

# inference
then = time.time()
depth = pipe(image, use_fast=True)["depth"]
depth.save("depth_image.png")
print(f"Took : {time.time()-then}")
image = Image.open("./pic.png")

then = time.time()
depth = pipe(image, use_fast=True)["depth"]
import ipdb;ipdb.set_trace()
depth.save("depth_imnage_2.png")
print(f"Took : {time.time() - then}")
