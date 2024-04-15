from PIL import Image
import numpy as np
from annotator.hed import HEDdetector
from annotator.uniformer import UniformerDetector

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