from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from PIL import Image
import os
from fastchat.model import get_conversation_template
from openai import OpenAI
import json
from utils import generate_sketch, encode_image, extract_json
from system_prompts import get_system_prompt
import argparse


def main(args):

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # initialize chat history
    system_prompt = get_system_prompt(args.max_tokens)
    conv = get_conversation_template(args.describer_model)
    conv.set_system_message(system_prompt)

    # flag for initial image reconstruction using ControlNet with sketch condition
    sketch_init = False

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

    for iteration in range(0, args.n_iter + 1):
        
        print(f"""\n{'='*36}\nIteration: {iteration}\n{'='*36}\n""")
        
        save_path = os.path.join(args.save_dir, f"iteration_{iteration}")
        os.makedirs(save_path, exist_ok=True)
        
        # initialization iteration: generate caption of source image X using GPT-4v
        if iteration == 0:

            # save source image
            X = Image.open(args.source_image_path)
            X_path = os.path.join(save_path, "x.png")
            X.save(X_path)

            # generate caption of source image using GPT-4v
            response = client.chat.completions.create(
                model=args.describer_model,
                messages=[
                    {
                    "role": "user",
                    "content": [
                        {
                        "type": "text",
                        "text": f"Provide a detailed caption of this image in {args.max_tokens} or less.",
                        },
                        {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encode_image(args.source_image_path)}"},
                        }
                    ],
                    }
                ],
                max_tokens=args.max_tokens,
            )
            prompt = response.choices[0].message.content
            
            # save response to json
            response_dict = {}
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
                                 generator=[torch.Generator(device="cuda").manual_seed(args.seed)],
                                 guidance_scale=9.0, 
                                 num_inference_steps=50).images
                
            else:
                # generate intial reconstruction using Stable Diffusion
                print(f"\nGenerating image...\n")
                images = sd_pipe(prompt=prompt, 
                                 generator=[torch.Generator(device="cuda").manual_seed(args.seed)],
                                 guidance_scale=8.0,
                                 num_inference_steps=50).images
                
            y = images[0]

        # GPT-4v receives last reconstruction and outputs a score, improvement, and prompt for next iteration
        else:
            
            # prompt separate judge to score similarity of the two images
            print("\nPrompting judge to provide similarity score...\n")
            score = client.chat.completions.create(
                model=args.describer_model,
                messages=[
                    {
                        "type": "text",
                        "text": "On a scale of 1-10, how similar are these two images? 1 means the two images are not similar at all and 10 means that they are identical."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encode_image(args.source_image_path)}"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encode_image(y_path)}"
                        }
                    },
                ]
            ).choices[0].message.content
            
            # get GPT-4v score, improvement, and updated response based on previous reconstruction and source image
            text = f"""
                   OBJECTIVE: Craft a prompt P where when P is entered to the image-to-image model, the model output image Y is as similar as possible to the source image X. \nSCORE: {score}
                """

            message=[
                {
                "type": "text",
                "text": text
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{encode_image(args.source_image_path)}"
                },
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{encode_image(y_path)}"
                },
                },
            ]

            # update chat history by appending GPT-4v prompt
            conv.append_message(conv.roles[0], message)
            
            # query GPT-4v with chat history and new prompt
            chat_history = conv.to_openai_api_messages()
            print(f"\nPrompting describer model to provide improved image prompt...\n")
            response = client.chat.completions.create(
                            model=args.describer_model,
                            messages=chat_history,
                            max_tokens=300,
                        )
            # update chat history to include GPT-4v response
            conv.append_message(conv.roles[1], response.choices[0].message.content)
            
            # extract improvement and prompt from GPT-4v response and append score to dictionary
            response_dict, json_str = extract_json(response.choices[0].message.content)
            response_dict["score"] = score
            response_path = os.path.join(save_path, 'response.json')
            with open(response_path, 'w') as f:
                json.dump(response_dict, f)
            print(f"\n[Score]:\n\n{response_dict["score"]}\n\n[Improvement]:\n\n\x1B[3m{response_dict["improvement"]}\x1B[0m\n\n[Prompt]:\n\n\x1B[3m{response_dict["prompt"]}\x1B[0m\n")

            # generate reconstruction using img2img model
            y_last = Image.open(os.path.join(args.save_dir, f"iteration_{iteration-1}", "y.png"))
            print(f"\nGenerating image...\n")
            images = i2i_pipe(prompt=response_dict['prompt'], 
                              image=y_last,
                              generator=[torch.Generator(device="cuda").manual_seed(args.seed)], 
                              strength=1.0, 
                              num_inference_steps=50, 
                              guidance_scale=9.0).images
            
            y = images[0]
        
        y_path = os.path.join(save_path, "y.png")
        y.save(y_path)
        print(f"\nReconstructed image saved at '{y_path}'\n")

        # Truncate conversation to avoid context length issues
        conv.messages = conv.messages[-2*(args.keep_last_n):]
            
    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--describer-model",
        type=str,
        default="gpt-4-turbo",
        help="Name of describer model."
    )
    parser.add_argument(
        "--source-image-path",
        type=str,
        default="/home/noah/IterativeRecon/examples/giraffe_original.jpg",
        help="Path to locally saved source image."
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        required=True,
        help="Path to local directory to save images and prompts."
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=20,
        help="Number of iterations to run."
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=77,
        help="Max number of tokens in GPT-4v generated prompts."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for diffusion model generator."
    )
    parser.add_argument(
        "--keep-last-n",
        type=int,
        default=3,
        help="Number of responses to save in chat history. If this is too large, then it may exceed the context window of the model."
    )
    args = parser.parse_args()

    main(args)

