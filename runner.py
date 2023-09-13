import asyncio 
from uuid import uuid4
import multiprocessing as mp 

from server import APIServer
from services.broker import ZMQBroker
from services.worker import ZMQWorker


from typing import List, Dict, Tuple, Any, Type

from log import logger 

import signal 

class ZMQRunner:
    def __init__(self):
        self.services:List[Tuple[str, Type[ZMQWorker], str, Dict[str, Any]], int] = []
        self.topics_acc:List[List[str]] = []
        self.processes:List[mp.Process] = []

    def __handle_termination_signal(self, signal_num:int, frame:str):
        logger.warning('runner has received the SIGTERM signal')
        logger.warning('runner will destroy all processes [broker] and [workers...]')
        for prs in self.processes:
            if prs.exitcode is None:
                prs.terminate()
                prs.join()
        
        signal.raise_signal(signal.SIGINT)

    def __initialize_signal_handler(self):
        signal.signal(
            signal.SIGTERM, 
            self.__handle_termination_signal
        )
        logger.info(f'runner has initialized the SIGNAL Handler')


    def add_service(self, name:str, topics:List[str], builder:Type[ZMQWorker], kwargs:Dict[str, Any], nb_workers:int):
        if not issubclass(builder, ZMQWorker):
            raise Exception(f'{builder} should be a subclass of ZMQWorker')
        
        nb_services = len(self.services)
        switch_id = f'switch_{nb_services:03d}'
        self.topics_acc.append(topics)
        self.services.append((name, builder, switch_id, kwargs, nb_workers))
    
    def start_server(self, server_config:Dict[str, str]):
        server_process = mp.Process(
            target=self.launch_server,
            kwargs=server_config
        )
        self.processes.append(
            server_process
        )

        self.processes[-1].start()

    def loop(self):
        broker_ = mp.Process(target=self.launch_broker)
        broker_.start()

        self.processes.append(broker_)
        for name, builder, switch_id, kwargs, nb_workers in self.services:
            for worker_id in range(nb_workers):
                worker_ = mp.Process(
                    target=self.launch_worker,
                    args=(builder, kwargs, f'{name}-{worker_id:03d}', switch_id)
                )
                self.processes.append(worker_)
                self.processes[-1].start()
        
        keep_loop = True 
        while keep_loop:
            try:
                process_states = [prs.exitcode is not None for prs in self.processes]
                if any(process_states):
                    keep_loop = False 
            except KeyboardInterrupt:
                logger.warning('runner will quit its loop')
                for prs in self.processes:
                    prs.join()
                keep_loop = False 
            except Exception as e:
                logger.error(e)
                keep_loop = False 
        
        logger.info('runner is waiting for worker to exit')
        for prs in self.processes:
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
    
    def launch_server(self, host:str, port:int, mounting_path:str ,path2base_dir:str, es_host:str, es_port:int, es_scheme:str, es_basic_auth:str, path2index_schema:str):
        async def start_server(host:str, port:int, path2base_dir:str, mounting_path:str):
            async with APIServer(host, port, path2base_dir, number_opened_file_limit=8192, mounting_path=mounting_path) as server:
                await server.loop(es_host, es_port, es_scheme, es_basic_auth, path2index_schema)
        
        asyncio.run(start_server(host, port, path2base_dir, mounting_path))


    def __enter__(self):
        self.__initialize_signal_handler()
    
        return self 
    
    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            logger.error(exc_value)
            logger.exception(traceback)
        logger.info('runner is out of its loop')




