import zmq 
import signal 
import threading

from log import logger 
from config import ZMQConfig

from time import sleep 
from typing import List, Tuple, Dict, Optional, Any

class ZMQBroker:
    def __init__(self, topics:List[List[str]]=[]):
        self.topics = topics 
        self.timeout = 1000
        self.setup_signal = True 
    
    def __handle_termination_signal(self, signal_num:int, frame:str):
        logger.warning('broker has received the SIGTERM signal')
        signal.raise_signal(signal.SIGINT)

    def __initialize_signal_handler(self):
        signal.signal(
            signal.SIGTERM, 
            self.__handle_termination_signal
        )
        logger.info(f'broker has initialized the SIGNAL Handler')

    def source(self, shared_mutex:threading.Lock, signal_tracker:threading.Event):
        router_socket = self.ctx.socket(zmq.ROUTER)
        router_socket.bind(ZMQConfig.CLIENT2BROKER_ADDRESS)

        pull_socket = self.ctx.socket(zmq.PULL)
        pull_socket.bind(ZMQConfig.WORKER2BROKER_ADDRESS)

        publisher_socket = self.ctx.socket(zmq.PUB)
        publisher_socket.bind(ZMQConfig.BROKER2WORKER_ADDRESS)

        poller = zmq.Poller()
        poller.register(pull_socket, zmq.POLLIN)
        poller.register(router_socket, zmq.POLLIN)
        
        logger.info('source is running')

        keep_loop = True 
        while keep_loop:
            try:
                mutex_was_acquired = shared_mutex.acquire(blocking=True, timeout=0.01)
                if mutex_was_acquired:
                    if signal_tracker.is_set():
                        logger.info('source has acquired the mutex and will exit')
                        keep_loop = False 
                    shared_mutex.release()

                map_socket2values = dict(poller.poll(timeout=self.timeout))
                if router_socket in map_socket2values:
                    if map_socket2values[router_socket] == zmq.POLLIN: 
                        client_id, _, topic, incoming_req = router_socket.recv_multipart()
                        logger.info(f'source has received message from {client_id} with the topic {topic}')
                    
                        publisher_socket.send(topic, flags=zmq.SNDMORE)
                        publisher_socket.send_multipart([client_id, incoming_req])

                if pull_socket in map_socket2values:
                    if map_socket2values[pull_socket] == zmq.POLLIN:
                        target_client_id, worker_response = pull_socket.recv_multipart()
                        logger.info(f'source has received the response for the client {target_client_id}')
                        router_socket.send_multipart([target_client_id, b'', worker_response])
            except Exception as e:
                logger.error(e) 
                keep_loop = False 
        # end while loop ...! 

        poller.unregister(pull_socket)
        poller.unregister(router_socket)
        
        pull_socket.close(linger=0)
        router_socket.close(linger=0)
        publisher_socket.close(linger=0)
        
        logger.info('source is out of its loop')

    def switch(self, switch_id:str, topics:List[str], shared_mutex:threading.Lock, signal_tracker:threading.Event):
        push_socket = self.ctx.socket(zmq.PUSH)
        push_socket.connect(ZMQConfig.WORKER2BROKER_ADDRESS)

        router_socket = self.ctx.socket(zmq.ROUTER)
        router_socket.bind(f'ipc://{switch_id}.ipc')

        subscriber_socket = self.ctx.socket(zmq.SUB)
        subscriber_socket.connect(ZMQConfig.BROKER2WORKER_ADDRESS)

        for topic in topics:
            subscriber_socket.setsockopt(zmq.SUBSCRIBE, topic.encode())
        
        poller = zmq.Poller()
        poller.register(router_socket, zmq.POLLIN)
        poller.register(subscriber_socket,zmq.POLLIN)

        logger.info(f'switch {switch_id} is running')
        workers = []
        keep_loop = True 
        while keep_loop:
            try:
                mutex_was_acquired = shared_mutex.acquire(blocking=True, timeout=0.01)
                if mutex_was_acquired:
                    if signal_tracker.is_set():
                        logger.info(f'switch {switch_id} has acquired the mutex and will exit')
                        keep_loop = False 
                    shared_mutex.release()

                logger.info(f'switch {switch_id} is running with {len(workers)} workers')
                map_socket2values = dict(poller.poll(timeout=self.timeout))
                if router_socket in map_socket2values:
                    if map_socket2values[router_socket] == zmq.POLLIN:
                        worker_id, _, worker_message_type, target_client_id, worker_response = router_socket.recv_multipart()
                        if worker_message_type == b'FREE':
                            workers.append(worker_id)
                        if worker_message_type == b'DATA':
                            push_socket.send_multipart([target_client_id, worker_response])
                
                if len(workers) > 0:
                    if subscriber_socket in map_socket2values:
                        if map_socket2values[subscriber_socket] == zmq.POLLIN:
                            
                            target_worker_id = workers.pop(0)  # FIFO
                            logger.info(f'switch {switch_id} has received the message from source and will route it to {target_worker_id}')
                            topic, client_id, incoming_req = subscriber_socket.recv_multipart()
                            router_socket.send_multipart([target_worker_id, b'', topic, client_id, incoming_req])
            except Exception as e:
                logger.error(e)
                keep_loop = False 

        poller.unregister(router_socket)
        poller.unregister(subscriber_socket)

        push_socket.close(linger=0)
        router_socket.close(linger=0)
        subscriber_socket.close(linger=0)

        logger.info(f'switch {switch_id} is out of its loop')

    def loop(self):
        shared_mutex = threading.Lock()
        signal_tracker = threading.Event()
        
        source_thread = threading.Thread(target=self.source, args=[shared_mutex, signal_tracker])
        source_thread.start()

        switch_threads:List[threading.Thread] = []
        switch_id = 0
        for topics_ in self.topics:
            switch_threads.append(
                threading.Thread(
                    target=self.switch,
                    args=[f'switch_{switch_id:03d}', topics_, shared_mutex, signal_tracker]
                )
            )
            switch_threads[-1].start()
            switch_id += 1
        
        keep_loop = True 
        while keep_loop:
            try:
                if not source_thread.is_alive():
                    keep_loop = False 
                if any([ not thread_.is_alive() for thread_ in switch_threads ]):
                    keep_loop = False 
                
                sleep(1)
            except KeyboardInterrupt:
                keep_loop = False 
                logger.info('broker has received SIGINT signal')
            except Exception as e:
                logger.error(e)
                keep_loop = False 
        
        with shared_mutex:
            signal_tracker.set()

        source_thread.join()
        for thread_ in switch_threads:
            thread_.join()

    def __enter__(self):    
        self.ctx = zmq.Context()
        if self.setup_signal:
            self.__initialize_signal_handler()
        
        return self 
    
    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            logger.error(exc_value)
            logger.exception(traceback)
        
        self.ctx.term()
        logger.warning('broker has finished to run')


if __name__ == '__main__':
    with ZMQBroker(100, True, [['IMG'], ['TXT'], ['SND', 'VID']]) as broker:
        broker.loop()