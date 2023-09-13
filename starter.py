
from runner import ZMQRunner

from vectorizer import NLPVectorizer, IMGVectorizer
from typing import Dict 

def launch_runner(nlp_model_name:str, img_model_name:str, nb_img_workers:int, nb_nlp_workers:int, cache_folder:str, device:str, chunk_size:int, server_config:Dict[str, str]):
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

        runner.start_server(server_config=server_config)
        runner.loop()

