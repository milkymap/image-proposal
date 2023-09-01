import numpy as np 
from services.worker import ZMQWorker
from sentence_transformers import SentenceTransformer

from libraries.strategies import load_tokenizer, split_text_into_chunks

from typing import List 
from io import BytesIO
from PIL import Image 

from time import sleep 

class NLPVectorizer(ZMQWorker):
    def __init__(self, model_name:str, cache_folder:str, device:str='cpu', chunk_size:int=128):
        super(NLPVectorizer, self).__init__()
        self.device = device 
        self.model = None #SentenceTransformer(model_name_or_path=model_name, cache_folder=cache_folder, device=self.device)
        self.tokenizer = load_tokenizer(encoding_name='gpt-3.5-turbo')
        self.chunk_size = chunk_size 

    def process_message(self, topic:bytes, incoming_req: bytes) -> np.ndarray:
        sleep(0.5)
        return True 
        text = incoming_req.decode()
        chunks:List[str] = split_text_into_chunks(text, chunk_size=self.chunk_size, tokenizer=self.tokenizer)
        embeddings:np.ndarray = self.model.encode(sentences=chunks, batch_size=32, device=self.device)
        
        if len(chunks) == 1:
            return embeddings[0].tolist()
        
        dot_scores = embeddings @ embeddings.T
        embedding_norms = np.linalg.norm(embeddings, axis=1)
        weighted_dot_scores = dot_scores / (embedding_norms[:, None] * embedding_norms[None, :])
        aggregated_dot_scores = np.sum(weighted_dot_scores, axis=1, keepdims=True)

        weighted_average_embedding = np.mean(aggregated_dot_scores * embeddings, axis=0)
        return weighted_average_embedding.tolist()

class IMGVectorizer(ZMQWorker):
    def __init__(self, model_name:str, cache_folder:str, device:str='cpu'):
        super(IMGVectorizer, self).__init__()
        self.device = device 
        self.model = None #SentenceTransformer(model_name_or_path=model_name, cache_folder=cache_folder, device=self.device)
        
    def process_message(self, topic:bytes, incoming_req: bytes) -> np.ndarray:
        sleep(0.5)
        return True 
        pil_image = Image.open(BytesIO(incoming_req))
        embedding = self.model.encode(sentences=pil_image, device=self.device)
        return embedding.tolist()


