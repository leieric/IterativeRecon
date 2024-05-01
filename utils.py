from PIL import Image
import numpy as np
from annotator.hed import HEDdetector
from annotator.uniformer import UniformerDetector
import base64
import logging
import ast

def generate_sketch(x: Image.Image, sketch_type: str) -> Image.Image:
    '''
    Generates sketch from input image

    Arguments:
        x: input image (PIL.Image.Image)
        sketch_type: type of sketch (str)
    
    Returns:
        sketch: sketch of input image (PIL.Image.Image)
    '''
    if sketch_type == 'hed':
        apply = HEDdetector()
        mode = 'L'
    elif sketch_type =='seg':
        apply = UniformerDetector()
        mode = 'L'
    else:
        raise ValueError("Not a valid sketch type. Choose 'hed' or 'seg'.")
    
    # convert image to numpy array
    x = np.array(x).astype(np.uint8)
    # generate sketch
    sketch = apply(x)
    # convert sketch back to PIL Image
    sketch = Image.fromarray(sketch, mode=mode)
    return sketch

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
        if not all(x in parsed for x in ["improvement","prompt"]):
            logging.error("Error in extracted structure. Missing keys.")
            logging.error(f"Extracted:\n {json_str}")
            return None, None
        return parsed, json_str
    except (SyntaxError, ValueError):
        logging.error("Error parsing extracted structure")
        logging.error(f"Extracted:\n {json_str}")
        return None, None