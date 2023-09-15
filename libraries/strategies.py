
import numpy as np 
import tiktoken

import operator as op 
import itertools as it, functools as ft 

from PIL import Image
from sentence_transformers import SentenceTransformer

from typing import List, Tuple, Dict, Any, Optional

def load_tokenizer(encoding_name:str='gpt-3.5-turbo') -> tiktoken.Encoding:
    return tiktoken.encoding_for_model(encoding_name)

def load_transformers(model_name:str, cache_folder:str, device:str='cpu') -> SentenceTransformer:
    return SentenceTransformer(
        model_name_or_path=model_name,
        cache_folder=cache_folder,
        device=device
    )

def split_text_into_chunks(text:str, chunk_size:int, tokenizer:tiktoken.Encoding) -> List[str]:
    tokens:List[int] = tokenizer.encode(text)
    nb_tokens = len(tokens)
    accumulator:List[str] = []
    for cursor in range(0, nb_tokens, chunk_size):
        paragraph = tokenizer.decode(tokens[cursor:cursor+chunk_size]) 
        accumulator.append(paragraph)
    
    return accumulator

def create_grid(images) -> Image:
    grid_size = (4, 4)
 
    image_width, image_height = images[0].size
    grid_width = image_width * grid_size[0]
    grid_height = image_height * grid_size[1]
 
    # Create a new blank image with the size of the grid
    grid_image = Image.new("RGB", (grid_width, grid_height))
 
    # Paste each image onto the grid
    for row in range(grid_size[1]):
        for col in range(grid_size[0]):
            index = row * grid_size[0] + col
            if index < len(images):
                grid_image.paste(images[index], (col * image_width, row * image_height))
 
    # Save or display the resulting grid image
    return grid_image