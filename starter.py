import asyncio 
from server import APIServer
from runner import ZMQRunner

from vectorizer import NLPVectorizer, IMGVectorizer

def launch_server(host:str, port:int, mounting_path:str ,path2base_dir:str, es_host:str, es_port:int, es_scheme:str, es_basic_auth:str, path2index_schema:str):
    async def start_server(host:str, port:int, path2base_dir:str, mounting_path:str):
        async with APIServer(host, port, path2base_dir, number_opened_file_limit=8192, mounting_path=mounting_path) as server:
            await server.loop(es_host, es_port, es_scheme, es_basic_auth, path2index_schema)
    
    asyncio.run(start_server(host, port, path2base_dir, mounting_path))

def launch_runner(nlp_model_name:str, img_model_name:str, nb_img_workers:int, nb_nlp_workers:int, cache_folder:str, device:str, chunk_size:int):
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

