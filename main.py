import click 
import torch as th 

from runner import ZMQRunner
from vectorizer import NLPVectorizer, IMGVectorizer

from log import logger 

from dotenv import load_dotenv

"""
python main.py --nlp_model_name Sahajtomar/french_semantic --img_model_name clip-ViT-L-14 --chunk_size 128 --timeout 1000 --nb_nlp_workers 1 --nb_img_workers 1
"""

@click.command()
@click.option('--host', default='0.0.0.0')
@click.option('--port', default=8000)
@click.option('--nlp_model_name', required=True)
@click.option('--img_model_name', required=True)
@click.option('--nb_nlp_workers', type=int, default=1)
@click.option('--nb_img_workers', type=int, default=1)
@click.option('--chunk_size', type=int, default=128)
@click.option('--timeout', type=int, default=100)
@click.option('--cache_folder', envvar='TRANSFORMERS_CACHE', required=True, type=click.Path(exists=True, file_okay=False))
@click.option('--protocol_buffers', envvar='PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', required=True, type=click.Choice(['python']))
def main(host:str, port:int, nlp_model_name:str, img_model_name:str, nb_nlp_workers:int, nb_img_workers:int ,chunk_size:int, timeout:int, cache_folder:str, protocol_buffers:str):
    device = 'cuda:0' if th.cuda.is_available() else 'cpu'
    th.multiprocessing.set_start_method('spawn')
    logger.info('multiprocessing => start_method set to spawn for cuda')

    with ZMQRunner() as runner:
        runner.add_service(
            name='img-embedding',
            topics=['IMAGE'],
            builder=IMGVectorizer,
            kwargs={'model_name': img_model_name, 'cache_folder': cache_folder, 'device': device},
            nb_workers=nb_img_workers
        )

        runner.add_service(
            name='txt-embedding',
            topics=['TEXT'],
            builder=NLPVectorizer,
            kwargs={'model_name': nlp_model_name, 'cache_folder': cache_folder, 'device': device, 'chunk_size': chunk_size},
            nb_workers=nb_nlp_workers
        )

        runner.loop()

    logger.info('... END ...')

if __name__ == '__main__':
    load_dotenv()
    main()
