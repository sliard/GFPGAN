import requests
import argparse
from PIL import Image
from io import BytesIO
from diffusers import LDMSuperResolutionPipeline
import torch

parser = argparse.ArgumentParser()

parser.add_argument(
    "--target",
    type=str,
    help="target file",
    default="ldm_generated_image.png"
)

parser.add_argument(
    "--input",
    type=str,
    help="input file url",
    default="https://user-images.githubusercontent.com/38061659/199705896-b48e17b8-b231-47cd-a270-4ffa5a93fa3e.png"
)
opt = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device :  {device}")
model_id = "CompVis/ldm-super-resolution-4x-openimages"

# load model and scheduler
pipeline = LDMSuperResolutionPipeline.from_pretrained(model_id)
pipeline = pipeline.to(device)

# let's download an  image
response = requests.get(opt.input)
low_res_img = Image.open(BytesIO(response.content)).convert("RGB")

# run pipeline in inference (sample random noise and denoise)
upscaled_image = pipeline(low_res_img, num_inference_steps=100, eta=1).images[0]
# save image
upscaled_image.save(opt.target)