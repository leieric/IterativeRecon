'''
Script for running human compression using stable diffusion models and human-user prompts.
'''

# libraries
import os
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
import torch

# parameters
MAX_ITER = 20


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
    dir = "/home/noah/IterativeRecon/examples/human_compress/"
    example_name = "giraffe"
    source_image_path = os.path.join(dir, example_name, "source.jpg")
    text_file_path = os.path.join(dir, example_name, "chat_history.txt")

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

            # obtain prompt from human-user
            prompt = input("\n\nEnter your prompt:\n\n")

            print("\n\nGenerating image:\n\n")

            # generate first reconstruction with stable diffusion
            images = sd_pipe(prompt=prompt, 
                             strength=0.7, 
                             num_inference_steps=50, 
                             guidance_scale=7.5).images

        else:

            print(f"\nLast generated image stored at {y_path}.\n")

            # obtain prompt from human-user
            prompt = input("\n\nEnter your prompt:\n\n")

            print("\n\nGenerating image:\n\n")

            # generate reconstruction with image-to-image model
            images = i2i_pipe(prompt=prompt, 
                               image=y, 
                               strength=0.7, 
                               num_inference_steps=50, 
                               guidance_scale=7.5).images
        
        # save reconstruction
        y = images[0]
        y_path = os.path.join(dir, example_name, f"recon_{iteration}.jpg")
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

