import click 
import torch as th 


from log import logger 

from dotenv import load_dotenv

from starter import launch_runner

@click.command()
@click.option('--host', default='0.0.0.0')
@click.option('--port', default=8000)
@click.option('--mounting_path', type=str, default='/')
@click.option('--path2base_dir', envvar='PATH2BASE_DIR', type=click.Path(exists=True, dir_okay=True, file_okay=False), required=True)

@click.option('--es_host', type=str, envvar='ES_HOST', required=True)
@click.option('--es_port', type=int, envvar='ES_PORT', required=True)
@click.option('--es_scheme', type=str, envvar='ES_SCHEME', required=True)
@click.option('--es_basic_auth', type=str, envvar='ES_BASIC_AUTH', default=None)
@click.option('--path2index_schema', type=click.Path(exists=True, dir_okay=False, file_okay=True), default='es_schema.json')

@click.option('--nlp_model_name', required=True)
@click.option('--img_model_name', required=True)
@click.option('--nb_nlp_workers', type=int, default=1)
@click.option('--nb_img_workers', type=int, default=1)
@click.option('--chunk_size', type=int, default=128)
@click.option('--cache_folder', envvar='TRANSFORMERS_CACHE', required=True, type=click.Path(exists=True, file_okay=False))
@click.option('--protocol_buffers', envvar='PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', required=True, type=click.Choice(['python']))
def main(host:str, port:int, mounting_path:str, path2base_dir:str, es_host:str, es_port:int, es_scheme:str, es_basic_auth:str, path2index_schema:str ,nlp_model_name:str, img_model_name:str, nb_nlp_workers:int, nb_img_workers:int ,chunk_size:int, cache_folder:str, protocol_buffers:str):
    
    device = 'cuda:0' if th.cuda.is_available() else 'cpu'
    if th.cuda.is_available():
        th.multiprocessing.set_start_method('spawn')
    
    logger.info('multiprocessing => start_method set to spawn for cuda')
    # move server into runner ...! 
    
    server_config = {
        'host': host,
        'port': port,
        'mounting_path': mounting_path,
        'path2base_dir': path2base_dir,
        'es_host': es_host,
        'es_port': es_port,
        'es_scheme': es_scheme,
        'es_basic_auth': es_basic_auth,
        'path2index_schema': path2index_schema
    }
    
    launch_runner(
        nlp_model_name=nlp_model_name,
        img_model_name=img_model_name,
        nb_nlp_workers=nb_nlp_workers,
        nb_img_workers=nb_img_workers,
        cache_folder=cache_folder,
        device=device,
        chunk_size=chunk_size,
        server_config=server_config
    )

    logger.info('... END ...')

if __name__ == '__main__':
    load_dotenv()
    main()
