import click 
import requests

from os import path 
from glob import glob 

import operator as op 
import itertools as it, functools as ft 

from tqdm import tqdm

from log import logger 

import asyncio 
import httpx


@click.command()
@click.option('--endpoint')
@click.option('--path2corpus')
def main(endpoint:str, path2corpus:str):

    img_extensions = ['*.jpg', '*.jpeg', '*.png']
    img_paths = []
    for ext in img_extensions:
        img_paths.extend(glob(path.join(path2corpus, ext)))
    
    sorted_img_paths = sorted(img_paths, key=lambda item: path.split(item)[-1])
    
    logger.info(f'nb_images : {len(sorted_img_paths)}')
    
    for img_path in tqdm(img_paths):
        extension = img_path.split('.')[-1]
        txt_path = img_path.replace(extension, 'txt')
        if not path.isfile(txt_path):
            exit(-1)

    for img_path in img_paths:
        extension = img_path.split('.')[-1]
        txt_path = img_path.replace(extension, 'txt')
        with open(txt_path, mode='r') as text_fp:
            # Open the image file for reading
            with open(img_path, "rb") as image_fp:
                files = {"image_file": (image_fp.name, image_fp), "text_file": (text_fp.name, text_fp)}
                response = requests.put(endpoint, files=files)
                if response.status_code == 200:
                    print(response.content)
                else:
                    print(txt_path)

        print('\n')

if __name__ == '__main__':
    main()