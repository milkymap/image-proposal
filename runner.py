import asyncio 
from uuid import uuid4
import multiprocessing as mp 

from server import APIServer
from services.broker import ZMQBroker
from services.worker import ZMQWorker
from vectorizer import NLPVectorizer, IMGVectorizer

from typing import List, Dict, Tuple, Any, Type

from log import logger 


class ZMQRunner:
    def __init__(self):
        self.services:List[Tuple[str, Type[ZMQWorker], str, Dict[str, Any]], int] = []
        self.topics_acc:List[List[str]] = []

    def add_service(self, name:str, topics:List[str], builder:Type[ZMQWorker], kwargs:Dict[str, Any], nb_workers:int):
        if not issubclass(builder, ZMQWorker):
            raise Exception(f'{builder} should be a subclass of ZMQWorker')
        nb_services = len(self.services)
        switch_id = f'switch_{nb_services:03d}'
        self.topics_acc.append(topics)
        self.services.append((name, builder, switch_id, kwargs, nb_workers))

    def loop(self):
        broker_ = mp.Process(target=self.launch_broker)
        broker_.start()

        processes = [broker_]
        for name, builder, switch_id, kwargs, nb_workers in self.services:
            for worker_id in range(nb_workers):
                worker_ = mp.Process(
                    target=self.launch_worker,
                    args=(builder, kwargs, f'{name}-{worker_id:03d}', switch_id)
                )
                processes.append(worker_)
                processes[-1].start()
        
        keep_loop = True 
        while keep_loop:
            try:
                process_states = [prs.exitcode is not None for prs in processes]
                if any(process_states):
                    keep_loop = False 
            except KeyboardInterrupt:
                for prs in processes:
                    prs.join()
            except Exception as e:
                keep_loop = False 

        for prs in processes:
            if prs.exitcode is None:
                prs.terminate()
                prs.join()

    def launch_broker(self):
        with ZMQBroker(topics=self.topics_acc) as broker_agent:
            broker_agent.loop()

    def launch_worker(self, builder:Type[ZMQWorker], kwargs:Dict[str, Any], worker_id:str, switch_id:str):
        with builder(**kwargs) as worker:
            worker.before_loop(worker_id, switch_id)
            worker.loop()

    def __enter__(self):
        return self 
    
    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            logger.error(exc_value)
            logger.exception(traceback)
        logger.info('runner is out of its loop')




