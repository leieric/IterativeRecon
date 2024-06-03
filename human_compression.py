'''
Script for running human compression using stable diffusion models and human-user prompts.
'''

# libraries
import os
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
import torch
from openai import OpenAI
from utils import encode_image

# parameters
MAX_ITER = 20
GPT_INIT = True
PROMPT_MODEL = 'gpt-4-vision-preview'
MAX_TOKENS = 77


def main():

    # model names
    sd_model_id = "stabilityai/stable-diffusion-2-1-base"
    device = "cuda"

    # load Stable Diffusion model
    sd_pipe = StableDiffusionPipeline.from_pretrained(sd_model_id, torch_dtype=torch.float16)
    sd_pipe = sd_pipe.to(device)

    # load Image-to-Image model
    i2i_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(sd_model_id, torch_dtype=torch.float16)
    i2i_pipe = i2i_pipe.to(device)

    # data directory, contains original source image 'source.jpg'
    # save_dir = "/home/noah/IterativeRecon/examples/human_compress/giraffe"
    save_dir = "results_human/"
    os.makedirs(save_dir, exist_ok=True)
    # source_image_path = "/home/noah/IterativeRecon/examples/giraffe_original.jpg"
    source_image_path = "/home/Shared/image_datasets/weissman/giraffe_original.jpg"
    text_file_path = os.path.join(save_dir, "chat_history.txt")

    os.makedirs(save_dir, exist_ok=True)
    os.system(f'cp {source_image_path} {os.path.join(save_dir, "source.jpg")}')
    with open(text_file_path, "w") as file:
        file.write("Chat History\n")

    # User Prompt
    user_prompt = f"""
    
    This is a conversation between you and an image-to-image generative model. 
    
    Your goal is to iteratively reconstruct a source image by prompting the image-to-image model to edit and refine the image outputs from your prompts.
                    
    Throughout the entire conversation you have access to the original source image, which is stored at {source_image_path}.
          
    In the initial iteration (Iteration 0), you will provide a prompt that you believe is a good starting point. The generative model will output an image that you can then view. 
    
    Att each iteration thereafter, you will view the generated image from the previous iteration and you will provide a prompt that the image-to-image model will use to edit and refine the generated image from the previous iteration.
          
    The objective is to generate the best possible reconstruction of the original source image. You have a total of {MAX_ITER} iterations to prompt the image-to-image model.
          
    At each iteration, you will have the option to finish early if you are satisfied with the reconstructed image at that iteration.
                    
    You may begin."""
    print(user_prompt)

    iteration = 0
    while iteration < MAX_ITER + 1:

        print(f"""\n{'='*36}\nIteration: {iteration}\n{'='*36}\n""")

        print(f"\nSource image stored at {source_image_path}.\n")

        # initial iteration
        if iteration == 0:

            # if GPT_INIT flag set True, obtain prompt from GPT-4v
            if GPT_INIT:
                
                print("\nGenerating initial prompt with GPT-4v...\n")

                client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
                response = client.chat.completions.create(
                    model=PROMPT_MODEL,
                    messages=[
                        {
                        "role": "user",
                        "content": [
                            {
                            "type": "text",
                            "text": f"Provide a detailed caption of this image in {MAX_TOKENS}, including only the most important visual details.",
                            },
                            {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpg;base64,{encode_image(source_image_path)}"},
                            }
                        ],
                        }
                    ],
                    max_tokens=MAX_TOKENS,
                )

                prompt = response.choices[0].message.content

                print(f"\n\x1B[3m{prompt}\x1B[0m\n")

            else:
                # obtain prompt from human-user
                prompt = input("\n\nEnter your prompt:\n\n")

            print("\n\nGenerating image:\n\n")

            # generate first reconstruction with stable diffusion
            images = sd_pipe(prompt=prompt,  
                             num_inference_steps=50, 
                             guidance_scale=9.0).images

        else:

            print(f"\nLast generated image stored at {y_path}.\n")

            # obtain prompt from human-user
            prompt = input("\n\nEnter your prompt:\n\n")

            print("\n\nGenerating image:\n\n")

            # generate reconstruction with image-to-image model
            images = i2i_pipe(prompt=prompt, 
                               image=y, 
                               strength=0.6, 
                               num_inference_steps=50, 
                               guidance_scale=7.5).images
        
        # save reconstruction
        y = images[0]
        y_path = os.path.join(save_dir, f"recon_{iteration}.jpg")
        y.save(y_path)
        print(f"\n\nReconstruction saved at {y_path}.\n\n")

        # save prompts to chat history
        with open(text_file_path, "a") as file:
            file.write(f"""\n{'='*36}\nIteration: {iteration}\n{'='*36}\n""")
            file.write(f"\n[Prompt]\n\n{prompt}\n")

        # ask user if they wish to continue
        term = input("\n\nDo you wish to continue? (y/n): ")
        while term not in ["y", "n"]:
            term = input("\n\nInvalid response. Do you wish to continue? (y/n): ")
        if term == "n":
            break
        
        iteration += 1

    return


if __name__ == "__main__":
    main()

