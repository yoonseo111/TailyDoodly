from diffusers import StableDiffusionPipeline
import torch
import os
import argparse
import json

model_id = "prompthero/openjourney"
pipe = StableDiffusionPipeline.from_pretrained(model_id)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
pipe = pipe.to(device)

def argument_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompts",
        type=str,
        help ="prompts json file path",
        default = "prompts.json"
    )
    parser.add_argument(
        "--output",
        type=str,
        help ="output image save path",
        default = "0_txt2img_output"
    )
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = argument_parse()
    with open(opt.prompts, 'r', encoding='utf-8') as file:
        data = json.load(file)
    output_dir = opt.output
    for p in range(len(data)) :
        prompt = data[p]
        image = pipe(prompt).images[0]
        image_name = str(p) + '.png'
        image_savepath = os.path.join(output_dir,image_name)
        image.save(image_savepath)
