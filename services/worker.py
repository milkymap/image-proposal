import zmq 
import signal 
import pickle 

from log import logger 
from config import ZMQConfig

from abc import ABC, abstractmethod
from typing import Optional, List, Tuple , Any 

class ZMQWorker(ABC):
    def __init__(self):
        self.timeout = 2000
        self.setup_signal = True 
    
    def before_loop(self, worker_id:str, switch_id:str):
        self.worker_id = worker_id
        self.switch_id = switch_id

        if self.setup_signal:
            self.__initialize_signal_handler()

        self.dealer_socket = self.ctx.socket(zmq.DEALER)
        self.dealer_socket.connect(f'ipc:///tmp/{self.switch_id}.ipc')

    def __handle_termination_signal(self, signal_num:int, frame:str):
        logger.warning(f'{self.worker_id} has received the SIGTERM signal')
        signal.raise_signal(signal.SIGINT)

    def __initialize_signal_handler(self):
        signal.signal(
            signal.SIGTERM, 
            self.__handle_termination_signal
        )
        logger.info(f'{self.worker_id} has initialized the SIGNAL Handler')

    def __consume_message(self, topic:bytes, incoming_req:bytes) -> Tuple[bool, Any]:
        try:
            return True, self.process_message(topic, incoming_req)
        except Exception as e:
            logger.error(e)
            return False, None 

    @abstractmethod
    def process_message(self, topic:bytes, incoming_req:bytes) -> Any:
        # strategy pattern based on topic
        pass 

    def loop(self):
        keep_loop = True 
        logger.info(f'{self.worker_id} is running')
        is_free = True 
        while keep_loop:
            try:          
                if is_free:
                    self.dealer_socket.send_multipart([b'', b'FREE', b'', b''])
                    is_free = False 

                if self.dealer_socket.poll(timeout=self.timeout) == zmq.POLLIN:
                    _, topic, client_id, incoming_req = self.dealer_socket.recv_multipart()
                    status, message_data = self.__consume_message(topic, incoming_req)
                    self.dealer_socket.send_multipart([b'', b'DATA', client_id], flags=zmq.SNDMORE)
                    self.dealer_socket.send_pyobj((status, message_data))
                    logger.info(f'{self.worker_id} has consumed the topic : {topic}')
                    is_free = True 
            except KeyboardInterrupt:
                logger.warning(f'{self.worker_id} has caught the SIGINT')
                keep_loop = False 
            except Exception as e:
                logger.error(e)
                keep_loop = False 
        
        logger.info(f'{self.worker_id} is out of its loop')
        
    def __enter__(self):
        self.ctx = zmq.Context()
        return self 
    
    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            logger.error(exc_value)
            logger.exception(traceback)

        self.dealer_socket.close(linger=0)
        self.ctx.term()
        logger.warning(f'{self.worker_id} has finished ro run')
                
