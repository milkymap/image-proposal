import zmq 
import asyncio
import zmq.asyncio as aiozmq 

from math import ceil 

import json 
import pickle 
import numpy as np 

from hashlib import sha256

import signal
import uvicorn

from io import BytesIO
from os import path 
from uuid import uuid4

from glob import glob 

from aiofile import async_open

from fastapi import FastAPI
from fastapi import BackgroundTasks, HTTPException, UploadFile, Form, File 
from fastapi.responses import JSONResponse, StreamingResponse

from apischema import ImageRecommendation, TaskStatus, MonitorContent, MonitorResponse, IndexCreateDeletion, VectorizeImageCorpus
from config import ZMQConfig
from log import logger 

from typing import List, Tuple, Dict, Optional, Any, Union

from elasticsearch import AsyncElasticsearch, exceptions

from PIL import Image 

import resource

class APIServer:
    __PATH2MEMORIES = 'map_task_id2task_data.pkl'
    def __init__(self, host:str, port:int, path2base_dir:str, number_opened_file_limit:int=8196, mounting_path:str="/"):
        self.host = host 
        self.port = port 
        self.path2base_dir = path2base_dir
        self.number_opened_file_limit = number_opened_file_limit
        self.mounting_path = mounting_path

        soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
        if number_opened_file_limit > soft_limit and number_opened_file_limit < hard_limit:
            logger.info(f'RLIMIT_NOFILE was set to {number_opened_file_limit}')
            resource.setrlimit(resource.RLIMIT_NOFILE, (number_opened_file_limit, hard_limit))
        
        soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
        logger.info(f'RLIMIT_NOFILE => {soft_limit}--{hard_limit}')

        self.api = FastAPI(
            docs_url='/',
            version="0.0.1",
            description="""
                Welcome to the API Server for Text and Image Vectorization!

                This API server empowers developers and data scientists to seamlessly integrate text and image
                vectorization into their applications and workflows. With a simple and intuitive interface, it offers
                a range of powerful features to enhance your data processing capabilities.

                Features:
                - Retrieve Images: Access images stored within the server's repository by making a simple API call.
                - Vectorize Text and Image Data: Effortlessly convert text and image data into high-dimensional vectors
                that are suitable for advanced machine learning and data analysis tasks.
                - Vectorize Image Corpora: Transform entire collections of images into vector representations for easy
                analysis and comparison.
                - Monitor Vectorization Progress: Keep track of the progress as the server vectorizes large image corpora,
                ensuring transparency and control over the process.
                - Make Image Recommendations: Leverage the power of vectorized data to generate accurate image
                recommendations based on input data.

                Whether you're building recommendation engines, content analysis tools, or any application that requires
                advanced data processing, this API server provides the essential tools to enhance your project's
                capabilities.

                Explore the various routes and endpoints provided by the server to start leveraging the benefits of
                text and image vectorization in your projects today.
            """
        ) 

        self.map_task_id2task_data:Dict[str, MonitorResponse] = {}
        self.api.add_event_handler('startup', self.handle_startup)
        self.api.add_event_handler('shutdown', self.handle_shutdown)
        self.define_routes()
    
    def define_routes(self):
        self.api.add_api_route('/heartbit', self.handle_heartbit)
        self.api.add_api_route('/get_image', self.handle_get_image)
        
        self.api.add_api_route('/create_index', self.handle_index_creation, methods=['POST'])
        self.api.add_api_route('/delete_index', self.handle_index_deletion, methods=['POST'])
        self.api.add_api_route('/inspect_index', self.handle_index_inspection)

        self.api.add_api_route('/add_text_image', self.handle_vectorize_text_image, methods=['POST'])
        self.api.add_api_route('/vectorize_image_corpus', self.handle_vectorize_corpus, methods=['POST'])
        self.api.add_api_route('/monitor_vectorize_corpus', self.monitor_vectorizer_corpus, response_model=MonitorResponse)
        self.api.add_api_route('/make_image_recommendation', self.handle_make_image_recommendation, methods=['POST'])
        
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
        self.global_shared_mutex = asyncio.Lock()
        self.ctx = aiozmq.Context() 

        if path.isfile(APIServer.__PATH2MEMORIES):
            logger.info(f'memories will be loaded from : {APIServer.__PATH2MEMORIES}')
            with open(APIServer.__PATH2MEMORIES, mode='rb') as fp:
                self.map_task_id2task_data:Dict[str, MonitorResponse] = pickle.loads(fp.read())

        try:
            self.elasticsearch_client = AsyncElasticsearch(**self.es_config)  # use qdrant async 
            await self.elasticsearch_client.__aenter__()  # open the es and initialize resources 
            es_info = await self.elasticsearch_client.info()
            logger.info(es_info)
        except Exception as e:
            logger.error(e)
            logger.info('SIGTERM will be raised')
            signal.raise_signal(signal.SIGTERM)
        else:
            logger.info('server has started its execution')

    async def handle_shutdown(self):
        with open(APIServer.__PATH2MEMORIES, mode='wb') as fp:
            pickle.dump(self.map_task_id2task_data, fp)
            logger.info(f'memories will be saved : {APIServer.__PATH2MEMORIES}')

        self.ctx.term()
        await self.elasticsearch_client.__aexit__()  # close the es and all resources 

        logger.info('server has stopped its running loop')

    async def loop(self, es_host:str, es_port:int, es_scheme:str, es_basic_auth:str, path2index_schema:str):
        if not path.isfile(path2index_schema):
            logger.error(f'{path2index_schema} is not a valid path')
            exit(-1)
        
        try:
            with open(path2index_schema, mode='r') as fp:
                self.es_index_schema = json.load(fp)
        except Exception as e:
            logger.error(e)
            exit(-1)

        keys_condition = 'text_image_pair' in self.es_index_schema and 'image' in self.es_index_schema
        if not keys_condition:
            logger.error('the index_schema is not valid')
            logger.error('the schema should contains config for : text_image_pair and image')
            exit(-1)

        self.es_config = {
            'hosts':[{
                'host': es_host,
                'port': es_port,
                'scheme': es_scheme 
            }],
            'basic_auth': es_basic_auth  
        }

        
        self.config = uvicorn.Config(app=self.api, host=self.host, port=self.port, root_path=self.mounting_path)
        self.server = uvicorn.Server(self.config)
        await self.server.serve()
    
    async def check_index(self, index_name:str) -> bool:
        text_image_pair_index_name = index_name + '_text_image_pair_index'
        image_index_name = index_name + '_image_index'

        try:
            index_states = await asyncio.gather(
                self.elasticsearch_client.indices.exists(index=text_image_pair_index_name),
                self.elasticsearch_client.indices.exists(index=image_index_name)
            )
        except Exception as e:
            logger.error(e)
            return False 

        return all(index_states)
        
    async def create_index(self, index_name:str, index_schema:Dict[str, Any]):
        index_status = await self.elasticsearch_client.indices.exists(index=index_name)
        if not index_status:
            logger.info(f'index : {index_name} will be created')
            index_creation_status = await self.elasticsearch_client.indices.create(
                index=index_name,
                mappings=index_schema['mappings'],
                settings=index_schema['settings']
            )
            logger.info(f'index creation status for {index_name} : {index_creation_status}')
            return True 

        logger.info(f'{index_name} was already created on the ES')
        logger.info(f'number of items: {index_name} => {await self.elasticsearch_client.count(index=index_name)}')
        return False  
    
    async def delete_index(self, index_name:str):
        index_status = await self.elasticsearch_client.indices.exists(index=index_name)
        if index_status:
            logger.info(f'index : {index_name} will be deleted')
            index_deletion_status = await self.elasticsearch_client.indices.delete(
                index=index_name
            )
            logger.info(f'index deletion status for {index_name} : {index_deletion_status}')
            return True 

        logger.info(f'{index_name} was not found in the ES')
        return False  
    
    async def handle_index_inspection(self, index_name:str):
        text_image_pair_index_name = index_name + '_text_image_pair_index'
        image_index_name = index_name + '_image_index'

        try:
            t_res = await self.elasticsearch_client.count(index=text_image_pair_index_name)
            i_res = await self.elasticsearch_client.count(index=image_index_name)
            
            return JSONResponse(
                status_code=200,
                content={
                    'index_name': index_name, 
                    'number_of_items_per_index': {
                        'text_image_pair_index': dict(t_res),
                        'image_index': dict(i_res)
                    }
                }
            )
        except Exception as e:
            logger.error(e)
            raise HTTPException(
                status_code=500,
                detail=f'failed to retrieve info of the index {index_name}'
            )

    async def handle_heartbit(self):
        return JSONResponse(
            status_code=200,
            content={
                'message': 'service is up and ready to process data'
            }
        )

    async def handle_get_image(self, dir_id:str, image_id:str):
        path2image = path.join(self.path2base_dir, dir_id, image_id)
        if path.isfile(path2image):
            try:
                pil_image = Image.open(path2image)
                binarystream_handler = BytesIO()
                pil_image.save(binarystream_handler, "JPEG")
                binarystream_handler.seek(0)  # rewind the file_pointer 
                return StreamingResponse(
                    binarystream_handler, 
                    media_type="image/jpeg"
                )
            except Exception as e:
                error_message = e 
                logger.error(error_message)
                raise HTTPException(
                    status_code=500,
                    detail=error_message
                )

        raise HTTPException(
            status_code=500,
            detail=f'{path2image} is not a valid path'
        )

    async def handle_index_creation(self, incoming_req:IndexCreateDeletion):
        text_image_pair_index_name = incoming_req.index_name + '_text_image_pair_index'
        image_index_name = incoming_req.index_name + '_image_index'

        fst = await self.create_index(text_image_pair_index_name, self.es_index_schema['text_image_pair'])
        snd = await self.create_index(image_index_name, self.es_index_schema['image'])

        if fst and snd:
            return JSONResponse(
                status_code=200,
                content=f'{incoming_req.index_name} was created successfully'
            )
        
        raise HTTPException(
            status_code=500,
            detail=f'failed to created the index {incoming_req.index_name}'
        )

    async def handle_index_deletion(self, incoming_req:IndexCreateDeletion):
        text_image_pair_index_name = incoming_req.index_name + '_text_image_pair_index'
        image_index_name = incoming_req.index_name + '_image_index'
        
        fst = await self.delete_index(text_image_pair_index_name)
        snd = await self.delete_index(image_index_name)

        if fst and snd:
            return JSONResponse(
                status_code=200,
                content=f'{incoming_req.index_name} was deleted successfully'
            )
        
        raise HTTPException(
            status_code=500,
            detail=f'failed to delete the index {incoming_req.index_name}'
        )

    async def perform_vector_search(self, input_vector:List[float], k:int, client: AsyncElasticsearch, index:str, reference:str ,source:List[str]=["id"], dir_ids:List[str]=None):
        task_id = str(uuid4())
        task = asyncio.current_task()
        task.set_name(f'background-task-{task_id}')

        query = {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": f"cosineSimilarity(params.queryVector, {reference}) + 1.0",
                    "params": {
                        "queryVector": input_vector
                    }
                }
            }
        }

        if dir_ids is not None and len(dir_ids) > 0:
            dir_id_filter_query = {
                "terms": {
                    "dir_id": dir_ids 
                }
            }

            query = {
                "bool": {
                    "filter": dir_id_filter_query,
                    "must": query  # Include the script_score query as a must condition
                }
            }

        try:
            response = await client.search(
                size=k,
                sort=[
                    {
                        "_score": {
                            "order": "desc"
                        }
                    }
                ],
                index=index,
                query=query,
                source=source 
            )
            return response
        except asyncio.CancelledError:
            pass 
        except exceptions.RequestError as e:
            logger.error(e)
            return None

    async def handle_make_image_recommendation(self, incoming_req: ImageRecommendation):
        # Generate a unique task ID and set the task name
        task_id = str(uuid4())
        task = asyncio.current_task()
        task.set_name(f'background-task-{task_id}')

        # Build index name 
        index_states = await self.check_index(incoming_req.index_name)
        if not index_states:
            raise HTTPException(
                status_code=500,
                detail=f'can not consume index {incoming_req.index_name}. check if this index is on the ES'
            )

        text_image_pair_index_name = incoming_req.index_name + '_text_image_pair_index'
        image_index_name = incoming_req.index_name + '_image_index'

        # Encode the incoming text for NLP processing
        txt_topic = b'TEXT'
        txt_binarystream = incoming_req.text.encode()

        # Call the NLP processing function
        nlp_response = await self.topic_handler(txt_topic, txt_binarystream)

        # Check if NLP processing was successful
        if not nlp_response[0]:
            raise HTTPException(status_code=500, detail='Failed to process text with NLP embedding model')

        nlp_embedding = nlp_response[1]

        # Perform vector search on the text-image pair index
        text_image_pair_search_response = await self.perform_vector_search(
            input_vector=nlp_embedding,
            k=incoming_req.text_nb_neighbors,
            client=self.elasticsearch_client,
            index=text_image_pair_index_name,
            reference="'text_vector'",
            source=['text', 'image_vector']
        )

        if text_image_pair_search_response is None:
            raise HTTPException(status_code=500, detail='Failed to search in text_image_pair index')

        # Extract image embeddings and scores from the search results
        hits = text_image_pair_search_response['hits']['hits']
        scores_acc = []
        img_embeddings_acc = []
        for hit in hits:
            img_embeddings_acc.append(hit['_source']['image_vector'])
            scores_acc.append(hit['_score'])

        if len(img_embeddings_acc) <= 1:
            raise HTTPException(status_code=500, detail='Number of retrieved neighbors (text-image-index) must be greater than 1')

        # Calculate the weighted average of image embeddings based on scores
        stacked_image_embedding = np.vstack(img_embeddings_acc)
        stacked_scores = np.array(scores_acc)[:, None]
        stacked_scores = stacked_scores / (np.sum(stacked_scores) + 1e-12)
        img_embedding = np.mean(stacked_image_embedding * stacked_scores, axis=0)

        # Perform vector search on the image index
        image_search_response = await self.perform_vector_search(
            input_vector=img_embedding,
            k=incoming_req.image_nb_neighbors,
            client=self.elasticsearch_client,
            index=image_index_name,
            reference="'vector'",
            source=["dir_id"],
            dir_ids=incoming_req.dir_ids
        )

        if image_search_response is None:
            raise HTTPException(status_code=500, detail='Failed to search in the image index')

        # Extract relevant information from the image search results
        hits = image_search_response['hits']['hits']
        image_ids = []
        for hit in hits:
            image_ids.append({
                'id': hit['_id'],
                'dir_id': hit['_source']['dir_id'],
                'score': hit['_score']
            })

        # Return the image IDs as JSON response
        return JSONResponse(status_code=200, content={'image_ids': image_ids})

    async def topic_handler(self, topic:bytes, binarystream:bytes) -> Tuple[bool, Optional[List[float]]]:
        task_id = str(uuid4())
        task = asyncio.current_task()
        task.set_name(f'background-task-topic-{task_id}')

        result = (False, None)

        dealer_socket:aiozmq.Socket = self.ctx.socket(zmq.DEALER)
        dealer_socket.connect(ZMQConfig.CLIENT2BROKER_ADDRESS)
            
        try:
            await dealer_socket.send_multipart([b'', topic, binarystream])
            _, resp = await asyncio.wait_for(dealer_socket.recv_multipart(), timeout=10)
            result = pickle.loads(resp)  # Tuple[bool, Optional[List[float]]]
        except asyncio.TimeoutError:
            logger.error(f'{task.get_name()} => has timedout ...!')
        except asyncio.CancelledError:
            logger.error(f'{task.get_name()} was cancelled') 
        except Exception as e:
            logger.error(f'{task.get_name()} => {e}')
        finally:
            dealer_socket.close(linger=0)

        return result
    
    async def __index_text_and_image(self, index_name:str, text:str, response) -> Union[JSONResponse, HTTPException]:
          # Build index name 
        text_image_pair_index_name = index_name + '_text_image_pair_index'
      
        nlp_response, img_response = response  # unpack the response 
        if nlp_response[0] == True and img_response[0] == True:
            nlp_embedding = nlp_response[1]
            img_embedding = img_response[1]
            document = {
                'text': text,
                'text_vector': nlp_embedding,
                'image_vector': img_embedding  # list of images in the next versions 
            }

            id = str(uuid4())
            try:
                insertion_response = await self.elasticsearch_client.index(
                    index=text_image_pair_index_name,
                    id=id,
                    document=document
                )
                return JSONResponse(
                    status_code=200,
                    content={
                        'id': id,
                    }
                )
            except Exception as e:
                error_message = e
                logger.error(f'{error_message}')
                return HTTPException(
                    status_code=500,
                    detail=error_message
                )
        
        return HTTPException(status_code=500, detail='failed to perform embedding on both image and text')

    async def handle_vectorize_text_image(self, index_name:str=Form(...), text:str=Form(...), image_file:UploadFile=File(...)):
        task_id = str(uuid4())
        task = asyncio.current_task()
        task.set_name(f'background-task-{task_id}')

        img_topic = b'IMAGE'
        txt_topic = b'TEXT'

        index_states = await self.check_index(index_name)  # gather is faster than sequential call 
        if not index_states:
            raise HTTPException(
                status_code=500,
                detail=f'can not consume index {index_name}. check if this index is on the ES'
            )

        try:
            txt_binarystream = text.encode()
            img_binarystream = await image_file.read()
        except Exception as e:
            error_message = e
            logger.error(f'{task.get_name()} => {error_message}')
            raise HTTPException(
                status_code=500,
                detail=error_message
            )

        awaitables = [self.topic_handler(txt_topic, txt_binarystream), self.topic_handler(img_topic, img_binarystream)]

        try:
            response = await asyncio.gather(*awaitables, return_exceptions=True)
        except asyncio.CancelledError:
            pass 
        except Exception as e:
            error_message = e
            logger.error(f'{task.get_name()} => {error_message}')
            raise HTTPException(
                status_code=500,
                detail=error_message
            )
        
        returned_val = await self.__index_text_and_image(index_name, text, response)
        if isinstance(returned_val, JSONResponse):
            return returned_val
        raise returned_val
    
    async def monitor_vectorizer_corpus(self, task_id:str):
        async with self.global_shared_mutex:
            task_data = self.map_task_id2task_data.get(task_id, MonitorResponse())
        return task_data

    async def handle_vectorize_corpus(self, incoming_req:VectorizeImageCorpus, background_tasks:BackgroundTasks):
        
        dir_id = incoming_req.dir_id
        index_name = incoming_req.index_name 
        concurrency = incoming_req.concurrency

        index_states = await self.check_index(index_name)
        if not index_states:
            raise HTTPException(
                status_code=500,
                detail=f'can not consume index {index_name}. check if this index is on the ES'
            )
        
        path2corpus = path.join(self.path2base_dir, dir_id)
        if path.isdir(path2corpus):
            task_id = sha256( (dir_id + index_name).encode() ).hexdigest()

            async with self.global_shared_mutex:
                task_data = self.map_task_id2task_data.get(task_id, MonitorResponse())
            
            if task_data.task_status == TaskStatus.UNDEFINED:
                background_tasks.add_task(
                    self.background_vectorize_corpus,
                    index_name,
                    dir_id, 
                    path2corpus,
                    concurrency,
                    task_id 
                )

                async with self.global_shared_mutex:
                    self.map_task_id2task_data[task_id] = MonitorResponse(
                        task_status=TaskStatus.PENDING
                    )

                return JSONResponse(status_code=200, content={'task_id': task_id})
            else:
                raise HTTPException(
                    status_code=500,
                    detail=json.dumps(task_data.dict())
                )
        raise HTTPException(
            status_code=500,
            detail=f"the path : {path2corpus} is not a valid directory"
        )
    
    async def synchronize_partitions(self, task_id:str, nb_files:int, nb_workers:int, shared_mutex:asyncio.Lock, signal_event:asyncio.Event):
        task = asyncio.current_task()
        task.set_name(f'background-task-{task_id}-synchronizer')
        
        pull_socket:aiozmq.Socket = self.ctx.socket(zmq.PULL)
        pull_socket.bind(f'inproc://{task_id}_progression')

        poller = aiozmq.Poller()
        poller.register(pull_socket, zmq.POLLIN)

        try:
            keep_looping = True 
            while keep_looping:
                try:
                    mutex_was_acquired = await asyncio.wait_for(shared_mutex.acquire(), timeout=0.01)
                    if mutex_was_acquired:
                        if signal_event.is_set():
                            keep_looping = False 
                    shared_mutex.release()
                except asyncio.TimeoutError:
                    pass 
                
                if not keep_looping:
                    break 

                logger.info(f'{task.get_name()} is running on the background')
                map_socket2value = dict(await poller.poll(timeout=1000))  
        
                if pull_socket in map_socket2value:
                    if map_socket2value[pull_socket] == zmq.POLLIN:
                        update_from_worker = await pull_socket.recv_json()
                        if update_from_worker['status'] == 'successful':
                            async with self.global_shared_mutex:
                                self.map_task_id2task_data[task_id].task_content.successful += update_from_worker['counter']
                        
                        if update_from_worker['status'] == 'failed':
                            async with self.global_shared_mutex:
                                self.map_task_id2task_data[task_id].task_content.failed += update_from_worker['counter']
                        
                        if update_from_worker['status'] == 'stopped':
                            nb_workers = nb_workers - 1

                keep_looping = nb_workers > 0 
            
            logger.info(f'{task.get_name()} is out of its loop')

        except asyncio.CancelledError:
            logger.info(f'{task.get_name()} was cancelled')
        except Exception as e:
            error_message = e
            logger.error(f'{task.get_name()} => {error_message}')
            async with shared_mutex:
                if not signal_event.is_set():
                    signal_event.set()

        poller.unregister(pull_socket)
        pull_socket.close(linger=0)

        logger.info(f'{task.get_name()} has closed its sockets')

    async def vectorize_partition(self, index_name:str, dir_id:str, partition:List[str], task_id:str, worker_id:str, shared_mutex:asyncio.Lock, signal_event:asyncio.Event):
        task = asyncio.current_task()
        task.set_name(f'background-task-{task_id}-worker:{worker_id}')

        # Build index name 
        image_index_name = index_name + '_image_index'
      
        dealer_socket:aiozmq.Socket = self.ctx.socket(zmq.DEALER)
        dealer_socket.connect(ZMQConfig.CLIENT2BROKER_ADDRESS)

        push_socket:aiozmq.Socket = self.ctx.socket(zmq.PUSH)
        push_socket.connect(f'inproc://{task_id}_progression')
    
        nb_files = len(partition)
        try:
            counter = 0
            topic = b'IMAGE'
            next_task = True 
            keep_processing = True 
            while counter < nb_files and keep_processing:      
                try:
                    mutex_was_acquired = await asyncio.wait_for(shared_mutex.acquire(), timeout=0.01)
                    if mutex_was_acquired:
                        if signal_event.is_set():
                            keep_processing = False 
                    shared_mutex.release()
                except asyncio.TimeoutError:
                    pass 
                
                if not keep_processing:
                    break 
                
                if next_task:
                    path2file = partition[counter]
                    logger.info(f'{task.get_name()} is running in the background {counter:03d}/{nb_files}')
                    async with async_open(path2file, mode='rb') as afp:
                        binarystream = await afp.read()
                    await dealer_socket.send_multipart([b'',topic, binarystream])
                    next_task = False 
                    
                incoming_signal = await dealer_socket.poll(timeout=100)
                if incoming_signal == zmq.POLLIN:
                    _, response = await dealer_socket.recv_multipart()  # make it non blocking call 
                    result = pickle.loads(response)  # Tuple[bool, Optional[List[float]]]
                    if result[0] == True:
                        _, filename = path.split(path2file)
                        insertion_response = await self.elasticsearch_client.index(
                            index=image_index_name,
                            id=filename,
                            document={
                                'dir_id': dir_id, 
                                'vector': result[1]  # 768 dims from clip-ViT-L-14
                            }
                        ) 
                        
                        await push_socket.send_json({
                            'status': 'successful',
                            'counter': 1  # add status failed | succedded for the monitoring 
                        })
                    else:
                        await push_socket.send_json({
                            'status': 'failed',
                            'counter': 1  # add status failed | succedded for the monitoring 
                        })
                    counter = counter + 1
                    next_task = True 

                    logger.info(f'{task.get_name()} is running >> {counter:03d}/{nb_files}')
                    
        except asyncio.CancelledError:
            logger.warning(f'{task.get_name()} was cancelled...!') 
        except Exception as e:
            error_message = e
            logger.error(f'{task.get_name()} => {error_message}')
            async with shared_mutex:
                if not signal_event.is_set():
                    signal_event.set()
        
        try:
            await asyncio.wait_for(
                push_socket.send_json({
                    'status': 'stopped',
                    'counter': 0  
                }),
                timeout=5
            )
        except asyncio.TimeoutError:
            pass 

        push_socket.close(linger=0)
        dealer_socket.close(linger=0)
        logger.info(f'{task.get_name()} has closed its sockets')

    async def background_vectorize_corpus(self, index_name:str, dir_id:str, path2corpus:str, concurrency:int, task_id:str):
        task = asyncio.current_task()
        task.set_name(f'background-task-{task_id}')
        logger.info(f'task {task_id} is waiting')
        extensions = ['*.jpg', '*.jpeg', '*.png']

        filepaths = []
        for ext in extensions:
            filepaths.extend(glob(path.join(path2corpus, ext)))

        nb_files = len(filepaths)
        batch_size = ceil(nb_files / concurrency)
        awaitables = []

        shared_mutex = asyncio.Lock()
        signal_event = asyncio.Event()

        async with self.global_shared_mutex:
            self.map_task_id2task_data[task_id] = MonitorResponse(
                task_status=TaskStatus.RUNNING,
                task_content=MonitorContent(
                    nb_files=nb_files,
                    failed=0,
                    successful=0
                )
            )

        try:            
            for cursor in range(0, nb_files, batch_size):
                partition = filepaths[cursor:cursor+batch_size]
                worker_id = f'{len(awaitables):03d}'
                awt = self.vectorize_partition(index_name, dir_id, partition, task_id, worker_id, shared_mutex, signal_event)
                awaitables.append(awt)

            nb_workers = len(awaitables)
            awaitables = [self.synchronize_partitions(task_id, nb_files, nb_workers, shared_mutex, signal_event)] + awaitables

            worker_responses = await asyncio.gather(*awaitables, return_exceptions=True)  
            async with shared_mutex:
                if signal_event.is_set():
                    async with self.global_shared_mutex: 
                        self.map_task_id2task_data[task_id].task_status = TaskStatus.FAILED
                else:
                    async with self.global_shared_mutex: 
                        self.map_task_id2task_data[task_id].task_status = TaskStatus.COMPLETED
                    
        except asyncio.CancelledError:
            logger.info(f'{task.get_name()} was cancelled')
            async with self.global_shared_mutex:
                self.map_task_id2task_data[task_id].task_status = TaskStatus.INTERRUPTED
        except Exception as e:
            logger.error(f'{task.get_name()} => {e}')
            async with self.global_shared_mutex:
                self.map_task_id2task_data[task_id].task_status = TaskStatus.FAILED
        
        logger.info(f'{task.get_name()} was done')

    async def __aenter__(self):
        return self 
    
    async def __aexit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            logger.error(exc_value)
            logger.exception(traceback)


    