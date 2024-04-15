from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from PIL import Image
import base64
import requests
import os
from fastchat.model import get_conversation_template
from openai import OpenAI
import json
import ast
import logging
from utils import generate_sketch

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# Function to extract dictionary object from openai completion object string
def extract_json(s):
    """
    Given an output from the attacker LLM, this function extracts the values
    for `improvement` and `adversarial prompt` and returns them as a dictionary.

    Args:
        s (str): The string containing the potential JSON structure.

    Returns:
        dict: A dictionary containing the extracted values.
        str: The cleaned JSON string.
    """
    # Extract the string that looks like a JSON
    start_pos = s.find("{") 
    end_pos = s.find("}") + 1  # +1 to include the closing brace
    if end_pos == -1:
        logging.error("Error extracting potential JSON structure")
        logging.error(f"Input:\n {s}")
        return None, None

    json_str = s[start_pos:end_pos]
    json_str = json_str.replace("\n", "")  # Remove all line breaks

    try:
        parsed = ast.literal_eval(json_str)
        if not all(x in parsed for x in ["score","improvement","prompt"]):
            logging.error("Error in extracted structure. Missing keys.")
            logging.error(f"Extracted:\n {json_str}")
            return None, None
        return parsed, json_str
    except (SyntaxError, ValueError):
        logging.error("Error parsing extracted structure")
        logging.error(f"Extracted:\n {json_str}")
        return None, None

MAX_TOKENS = 77
N_ITERATIONS = 20
PROMPT_MODEL = 'gpt-4-vision-preview'
SOURCE_PATH = "/home/noah/Text-Sketch/recon_examples/PICS/finetune/Cheng2020_lmbda1.0/CLIC2021/source/18.png"
SOURCE_IMG = encode_image(SOURCE_PATH)
SAVE_DIR = "./logs/source18"

SYSTEM_PROMPT = f"""You are a helpful assistant. You must obey and follow all of the following instructions.
    
                SETTING:
                This is a chat between you and an image-to-image model, where you provide a prompt P and receive the image-to-image model's output Y as a response. The image-to-image model receives only your prompt P and the previous image that the image-to-image model generated Y**. The image-to-image model has no knowledge of the chat history. You will receive a source image X at the beginning of the conversation. **Your objective is to craft a prompt P where when P is entered to the image-to-image model, the model outputs an image Y as similar to the source image X as possible. The prompt P must be {MAX_TOKENS} tokens or less.** Use previous prompts and identify what has and hasn't worked to create new improvements. 

                FORMAT:
                Format your response in JSON, with the following elements: `score`, `improvement`, and `prompt`. The `score` value contains an integer value from 1 to 10 evaluating how similar the image-to-image model's output Y is to the source image X, where 1 means Y and X are not similar at all and 10 means Y and X are very similar. The `improvement` value contains a few sentences describing the similarities and differences between the image-to-image model's output Y and the source image X, and how the prompt should be modified to achieve the goal. The `prompt` value contains the new prompt P that will be entered to the image-to-image model and it must be {MAX_TOKENS} tokens or less. Use the ideas listed in `improvement` and previous prompts to improve and refine your new prompt.  Your response should **only** contain this JSON element and nothing else. Each of your responses is a single refinement of P. When proposing a refinement of a prompt P, do not repeat the previous prompt, and instead propose new changes and improvements. 
                
                The user output you receive is composed of only the image-to-image model's output Y.
                """

INIT_MESSAGE = "Your objective is to craft a prompt P where when P is entered to an image-to-image model, the model output image Y is as similar as possible to the source image X. Begin."

def main():

    # conv = get_conversation_template(PROMPT_MODEL)
    # conv.set_system_message(SYSTEM_PROMPT)

    # flag for initial image reconstruction using ControlNet with sketch condition
    sketch_init = True

    sd_model_id = "stabilityai/stable-diffusion-2-1-base"
    cn_model_id = "thibaud/controlnet-sd21-hed-diffusers"
    device = "cuda"

    # load Stable Diffusion model
    sd_pipe = StableDiffusionPipeline.from_pretrained(sd_model_id, torch_dtype=torch.float16)
    sd_pipe = sd_pipe.to(device)

    # load Image-to-Image model
    i2i_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(sd_model_id, torch_dtype=torch.float16)
    i2i_pipe = i2i_pipe.to(device)

    # load ControlNet model
    controlnet = ControlNetModel.from_pretrained(cn_model_id, torch_dtype=torch.float16)
    cn_pipe = StableDiffusionControlNetPipeline.from_pretrained(sd_model_id, 
                                                                controlnet=controlnet, 
                                                                torch_dtype=torch.float16, 
                                                                variant="fp16")
    cn_pipe.scheduler = UniPCMultistepScheduler.from_config(cn_pipe.scheduler.config)
    cn_pipe = cn_pipe.to(device)

    for iteration in range(0, N_ITERATIONS + 1):
        
        print(f"""\n{'='*36}\nIteration: {iteration}\n{'='*36}\n""")
        
        save_path = os.path.join(SAVE_DIR, f"iteration_{iteration}")
        os.makedirs(save_path, exist_ok=True)
        
        # initialization iteration: generate caption of source image X using GPT-4v, HED map of source image, and pass as input to ControlNet
        response_dict = {}
        if iteration == 0:

            # save source image
            X = Image.open(SOURCE_PATH)
            X_path = os.path.join(save_path, "x.png")
            X.save(X_path)

            # generate caption of source image using GPT-4v
            response = prompter(iter=iteration, init=True)
            prompt = response.choices[0].message.content
            
            # save response to json
            response_dict['prompt'] = prompt
            response_path = os.path.join(save_path, 'response.json')
            with open(response_path, 'w') as f:
                json.dump(response_dict, f)
            
            print(f"\n[Prompt]:\n\n\x1B[3m{prompt}\x1B[0m\n")

            if sketch_init:
                # generate initial reconstruction using ControlNet
                source_sketch = generate_sketch(x=X, sketch_type='hed')
                sketch_path = os.path.join(save_path, "sketch.png")
                source_sketch.save(sketch_path)
                print(f"\nSketch saved at '{sketch_path}'\n")
                
                print(f"\nGenerating image...\n")
                images = cn_pipe(prompt=prompt, 
                            image=source_sketch,
                            guidance_scale=9.0, 
                            num_inference_steps=50).images
                
            else:
                # generate intial reconstruction using Stable Diffusion
                print(f"\nGenerating image...\n")
                images = sd_pipe(prompt=prompt, 
                                 strength=0.6, 
                                 guidance_scale=7.5).images
                
            y = images[0]

        # GPT-4v receives last reconstruction and outputs a score, improvement, and prompt for next iteration
        else:
            # get GPT-4v score, improvement, and updated response based on previous reconstruction and source image
            response = prompter(iter=iteration, init=False)
            response_dict, json_str = extract_json(response.choices[0].message.content)
            response_path = os.path.join(save_path, 'response.json')
            with open(response_path, 'w') as f:
                json.dump(response_dict, f)
            print(f"\n[Score]:\n\n{response_dict["score"]}\n\n[Improvement]:\n\n\x1B[3m{response_dict["improvement"]}\x1B[0m\n\n[Prompt]:\n\n\x1B[3m{response_dict["prompt"]}\x1B[0m\n")

            # generate reconstruction using img2img model
            y_last = Image.open(os.path.join(SAVE_DIR, f"iteration_{iteration-1}", "y.png"))
            print(f"\nGenerating image...\n")
            images = i2i_pipe(prompt=prompt, image=y_last, strength=0.5, num_inference_steps=50, guidance_scale=7.5).images
            
            y = images[0]
        
        y_path = os.path.join(save_path, "y.png")
        y.save(y_path)
        print(f"\nReconstructed image saved at '{y_path}'\n")
            
    return

def prompter(iter: int, init=False):
    """
    Generates a caption of the provided image using GPT-4v

    Parameters
    -   iter: current iteration number (int)
    -   init: boolean flag, if True generates a caption of the input image with GPT-4v
                            if False generates a json object response including a similarity score,
                            improvement, and new prompt

    Returns
    -   response: GPT-4v response (ChatCompletion object)
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # if initialization, generate image prompt with GPT-4v
    if init:
        response = client.chat.completions.create(
            model=PROMPT_MODEL,
            messages=[
                {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": f"Provide a detailed caption of this image in {MAX_TOKENS} or less.",
                    },
                    {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{SOURCE_IMG}"},
                    }
                ],
                }
            ],
            max_tokens=MAX_TOKENS,
        )

    # generate similarity score, improvement, and new prompt with GPT-4v
    else:
        # load and encode last reconstruction Y
        y_path = os.path.join(SAVE_DIR, f"iteration_{iter-1}", "y.png")
        y_img = encode_image(y_path)

        response = client.chat.completions.create(
            model=PROMPT_MODEL,
            messages=[
                {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    },
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{SOURCE_IMG}"
                    },
                    },
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{y_img}"
                    },
                    },
                ],
                }
            ],
            max_tokens=300,
            )

    return response


if __name__ == "__main__":
    main()


# def generator(img: Image.Image, prompt: str, init=False) -> Image.Image:
#     """
#     Generates reconstruction of source image using Stable Diffusion Img2Img model
#     with previous reconstructed image and update prompt caption.

#     Parameters:
#     -   img: image input to Img2Img model (Image.Image object)
#     -   prompt: prompt input to Img2Img model (string)
#     -   init: boolean flag, if True runs text-to-image stable diffusion model,
#                             if False runs image-to-image stable diffusion with
#                             passed img 

#     Returns:
#     -   recon: output image from Img2Img model (Image.Image object)
#     """
#     # set device to cuda and load pretrained Img2Img model
#     device = "cuda"
#     model_id_or_path = "runwayml/stable-diffusion-v1-5"

#     # if init flag is true, run stable diffusion from initial caption
#     if init:
#         sd_pipe = StableDiffusionPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
#         sd_pipe = sd_pipe.to(device)
#         images = sd_pipe(prompt=prompt, strength=0.6, guidance_scale=7.5).images
#         recon = images[0]
    
#     else:
#         i2i_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
#         i2i_pipe = i2i_pipe.to(device)
#         images = i2i_pipe(prompt=prompt, image=img, strength=0.6, guidance_scale=7.5).images
#         recon = images[0]

#     return recon