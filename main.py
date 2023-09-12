import click 
import torch as th 

from runner import ZMQRunner

from log import logger 

from dotenv import load_dotenv
from time import sleep 

import multiprocessing as mp 
from starter import launch_runner, launch_server

@click.command()
@click.option('--host', default='0.0.0.0')
@click.option('--port', default=8000)
@click.option('--path2base_dir', envvar='PATH2BASE_DIR', type=click.Path(exists=True, dir_okay=True, file_okay=False), required=True)

@click.option('--es_host', type=str, envvar='ES_HOST', required=True)
@click.option('--es_port', type=int, envvar='ES_PORT', required=True)
@click.option('--es_scheme', type=str, envvar='ES_SCHEME', required=True)
@click.option('--es_basic_auth', type=str, envvar='ES_BASIC_AUTH', required=True)
@click.option('--path2index_schema', type=click.Path(exists=True, dir_okay=False, file_okay=True), default='es_schema.json')

@click.option('--nlp_model_name', required=True)
@click.option('--img_model_name', required=True)
@click.option('--nb_nlp_workers', type=int, default=1)
@click.option('--nb_img_workers', type=int, default=1)
@click.option('--chunk_size', type=int, default=128)
@click.option('--cache_folder', envvar='TRANSFORMERS_CACHE', required=True, type=click.Path(exists=True, file_okay=False))
@click.option('--protocol_buffers', envvar='PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', required=True, type=click.Choice(['python']))
def main(host:str, port:int, path2base_dir:str, es_host:str, es_port:int, es_scheme:str, es_basic_auth:str, path2index_schema:str ,nlp_model_name:str, img_model_name:str, nb_nlp_workers:int, nb_img_workers:int ,chunk_size:int, cache_folder:str, protocol_buffers:str):
    device = 'cuda:0' if th.cuda.is_available() else 'cpu'
    if device == 'cuda:0':
        th.multiprocessing.set_start_method('spawn')
        logger.info('multiprocessing => start_method set to spawn for cuda')

    runner_process = mp.Process(
        target=launch_runner,
        kwargs = {
            'nlp_model_name': nlp_model_name,
            'img_model_name': img_model_name,
            'nb_img_workers': nb_img_workers,
            'nb_nlp_workers': nb_nlp_workers,
            'cache_folder': cache_folder,
            'device': device,
            'chunk_size': chunk_size
        }
    )

    server_process = mp.Process(
        target=launch_server,
        kwargs = {
            'host': host,
            'port': port,
            'path2base_dir': path2base_dir,
            'es_host': es_host,
            'es_port': es_port,
            'es_scheme': es_scheme,
            'es_basic_auth': es_basic_auth,
            'path2index_schema': path2index_schema
        }
    )
    try:
        runner_process.start()
    except Exception as e:
        logger.error(e)
        exit(-1)
    try:
        server_process.start()
    except Exception as e:
        logger.error(e)
        runner_process.terminate()
        runner_process.join()
        exit(-1)

    processes_tracker = [runner_process, server_process]

    keep_monitoring = True 
    while keep_monitoring:
        try:
            if any([ prs.exitcode is not None for prs in processes_tracker ]):
                keep_monitoring = False 
            sleep(1)
        except KeyboardInterrupt:
            for prs in processes_tracker:
                prs.join()  # wait for prs to get the SIGINT signal and to exit
        except Exception as e:
            logger.error(e)
    
    for prs in processes_tracker:
        if prs.exitcode is not None:
            prs.terminate()  # send SIGTERM to prs 
            prs.join()  # wait for prs to receive the SIGTERM and to exit 


    logger.info('... END ...')

if __name__ == '__main__':
    load_dotenv()
    main()
