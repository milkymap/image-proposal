import zmq 
import asyncio
import zmq.asyncio as aiozmq 

import pickle 
import numpy as np 

import signal
import uvicorn

from os import path 
from uuid import uuid4

from glob import glob 
from time import sleep 

from aiofile import async_open

from fastapi import FastAPI
from fastapi import BackgroundTasks, HTTPException, UploadFile, Form, File 
from fastapi.responses import JSONResponse

from apischema import ImageRecommendation
from config import ZMQConfig, TaskStatus
from log import logger 

from typing import List, Tuple, Dict, Optional

from elasticsearch import AsyncElasticsearch

class APIServer:
    __PATH2MEMORIES = 'map_task_id2task_status.pkl'
    def __init__(self, host:str, port:int, max_sockets:int=1024):
        self.host = host 
        self.port = port 
        self.max_sockets = max_sockets

        self.api = FastAPI(
            version="0.0.1",
            description="""

            """
        ) 

        self.map_task_id2task_status:Dict[str, TaskStatus] = {}
        self.api.add_event_handler('startup', self.handle_startup)
        self.api.add_event_handler('shutdown', self.handle_shutdown)
        self.define_routes()
    
    def define_routes(self):
        self.api.add_api_route('/vectorize_text_image', self.handle_vectorize_text_image, methods=['POST'])
        self.api.add_api_route('/vectorize_image_corpus', self.handle_vectorize_corpus)
        self.api.add_api_route('/monitor_vectorize_corpus', self.monitor_vectorizer_corpus)
        #self.api.add_api_route('/make_image_recommendation', self.handle_make_image_recommendation)
        
    async def __handle_sigterm(self):
        logger.info('SIGTERM was caught, SIGINT will be raised')
        signal.raise_signal(signal.SIGINT)

    async def __remove_resources(self):
        logger.info('SIGINT was caught ...!')
        tasks = [task for task in asyncio.all_tasks() if task.get_name().startswith('background-task')]
        for task in tasks:
            task.cancel()
            
        await asyncio.gather(*tasks)
        logger.info('all background tasks were removed')
        self.server.should_exit = True 
        
    async def handle_startup(self):
        self.event_loop = asyncio.get_running_loop()
        self.event_loop.add_signal_handler(
            signal.SIGTERM,
            callback=lambda: asyncio.create_task(self.__handle_sigterm())
        )
        self.event_loop.add_signal_handler(
            signal.SIGINT,
            callback=lambda: asyncio.create_task(self.__remove_resources())
        )
        self.mutex = asyncio.Lock()
        self.semaphore = asyncio.Semaphore(value=self.max_sockets)
        self.ctx = aiozmq.Context() 

        if path.isfile(APIServer.__PATH2MEMORIES):
            with open(APIServer.__PATH2MEMORIES, mode='rb') as fp:
                self.map_task_id2task_status = pickle.loads(fp.read())

        self.elasticsearch_client = AsyncElasticsearch(hosts=["http://localhost:9200"])  # use qdrant async 
        print(await self.elasticsearch_client.info())

        logger.info('server has started its execution')

    async def handle_shutdown(self):
        with open(APIServer.__PATH2MEMORIES, mode='wb') as fp:
            pickle.dump(self.map_task_id2task_status, fp)

        self.ctx.term()
        logger.info('server has stopped its running loop')

    async def loop(self):
        self.config = uvicorn.Config(app=self.api, host=self.host, port=self.port)
        self.server = uvicorn.Server(self.config)
        await self.server.serve()
    
    async def __compute_embedding(self, topic:bytes, binarystream:bytes) -> Tuple[bool, Optional[np.ndarray]]:
        task_id = str(uuid4())
        task = asyncio.current_task()
        task.set_name(f'background-task-{task_id}')

        result = (False, None)
              
        try:
            async with self.semaphore:
                dealer_socket:aiozmq.Socket = self.ctx.socket(zmq.DEALER)
                dealer_socket.connect(ZMQConfig.CLIENT2BROKER_ADDRESS)
                try:
                    await dealer_socket.send_multipart([b'', topic, binarystream])
                    _, resp = await dealer_socket.recv_multipart()
                    result = pickle.loads(resp)
                except Exception as e:
                    logger.error(e)
                else:
                    pass 
                finally:
                    dealer_socket.close(linger=0)
        except asyncio.CancelledError:
            pass 
        
        return result
    
    async def handle_vectorize_text_image(self, article:str=Form(...), image_file:UploadFile=File(...)):
        img_topic = b'IMAGE'
        txt_topic = b"TEXT"

        txt_binarystream = article.encode()
        img_binarystream = await image_file.read()

        awaitables = [
            self.__compute_embedding(txt_topic, txt_binarystream),
            self.__compute_embedding(img_topic, img_binarystream)
        ]

        response = await asyncio.gather(*awaitables, return_exceptions=True)
        # save the response into the index
        return response 

    async def monitor_vectorizer_corpus(self, task_id:str):
        pair_socket:aiozmq.Socket = self.ctx.socket(zmq.PAIR)
        pair_socket.connect(f'inproc://{task_id}')

        async with self.mutex:
            task_status = self.map_task_id2task_status.get(task_id, TaskStatus.UNDEFINED)
        
        if task_status == TaskStatus.RUNNING:
            await pair_socket.send(b'...')
            task_progression = await pair_socket.recv_json()
        else:
            task_progression = None 
        
        pair_socket.close(linger=0)

        return JSONResponse(
            status_code=200,
            content={
                'task_status': task_status,
                'task_progression': task_progression
            }
        )

    async def handle_vectorize_corpus(self, path2corpus:str, background_tasks:BackgroundTasks):
        if path.isdir(path2corpus):
            task_id = str(uuid4())
            background_tasks.add_task(
                self.background_vectorize_corpus,
                path2corpus,
                task_id 
            )
            return JSONResponse(status_code=200, content={'task_id': task_id})
        raise HTTPException(
            status_code=500,
            detail=f"{path2corpus} is not a valid directory"
        )
        
    async def background_vectorize_corpus(self, path2corpus:str, task_id:str):
        task = asyncio.current_task()
        task.set_name(f'background-task-{task_id}')
        logger.info(f'{task_id} is waiting')
        async with self.mutex:
            self.map_task_id2task_status[task_id] = TaskStatus.PENDING

        try:
            async with self.semaphore:   
                async with self.mutex:
                    self.map_task_id2task_status[task_id] = TaskStatus.RUNNING

                pair_socket:aiozmq.Socket = self.ctx.socket(zmq.PAIR)
                pair_socket.bind(f'inproc://{task_id}')

                dealer_socket:aiozmq.Socket = self.ctx.socket(zmq.DEALER)
                dealer_socket.connect(ZMQConfig.CLIENT2BROKER_ADDRESS)
                try:
                    topic = b'IMAGE'
                    filepaths = glob(path.join(path2corpus, '*'))
                    nb_files = len(filepaths)
                    counter = 0
                    for path2file in filepaths:
                        polling_value = await pair_socket.poll(timeout=10)  # 10ms 
                        if polling_value == zmq.POLLIN: 
                            tiktak = await pair_socket.recv()
                            if tiktak == b'...':
                                await pair_socket.send_json({
                                    'nb_files': nb_files,
                                    'progression': counter 
                                })

                        logger.info(f'{task_id} vectorize is running in the background {counter:03d}/{len(filepaths)}')
                        async with async_open(path2file, mode='rb') as afp:
                            binarystream = await afp.read()
                        await dealer_socket.send_multipart([b'',topic, binarystream])
                        _, response = await dealer_socket.recv_multipart()
                        print(response[:10], counter)  # save the response into the index ES
                        counter = counter + 1
                except asyncio.CancelledError:
                    async with self.mutex:
                        self.map_task_id2task_status[task_id] = TaskStatus.INTERRUPTED
                    logger.info(f'{task_id} was cancelled by the SIGINT handler')
                except Exception as e:
                    logger.error(e)
                    async with self.mutex:
                        self.map_task_id2task_status[task_id] = TaskStatus.FAILED 
                else:
                    logger.info(f'{task_id} has terminated')
                    async with self.mutex:
                        self.map_task_id2task_status[task_id] = TaskStatus.COMPLETED
                finally:
                    pair_socket.close(linger=0)
                    dealer_socket.close(linger=0)
                    logger.info(f'{task_id} has closed its sockets')
        except asyncio.CancelledError:
            logger.info(f'{task_id} has caugut the CANCELLED signal before entering into the semaphore')
            async with self.mutex:
                self.map_task_id2task_status[task_id] = TaskStatus.INTERRUPTED
        
        logger.info(f'{task_id} ... is out of its loop')

    async def __aenter__(self):
        return self 
    
    async def __aexit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            logger.error(exc_value)
            logger.exception(traceback)


def launch_server(host:str, port:int):
    async def start_server(host:str, port:int):
        async with APIServer(host, port) as server:
            await server.loop()
    
    asyncio.run(start_server(host, port))
    