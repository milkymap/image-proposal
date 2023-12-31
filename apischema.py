from pydantic import BaseModel
from enum import Enum

from typing import Optional, List 

class ImageRecommendation(BaseModel):
    text:str 
    index_name:str 
    text_nb_neighbors:int=7   
    image_nb_neighbors:int=16
    dir_ids:Optional[List[str]]=None 

class TaskStatus(str, Enum):
    FAILED:str="FAILED"
    UNDEFINED:str="UNDEFINED"

    PENDING:str="PENDING"
    RUNNING:str="RUNNING"

    TIMEDOUT:str="TIMEDOUT"
    COMPLETED:str="COMPLETED"
    INTERRUPTED:str="INTERRUPTED"

class MonitorContent(BaseModel):
    nb_files:int 
    failed:int 
    successful:int 

class MonitorResponse(BaseModel):
    task_status:TaskStatus=TaskStatus.UNDEFINED
    task_content:Optional[MonitorContent]=None 

class IndexCreateDeletion(BaseModel):
    index_name:str 

class VectorizeImageCorpus(BaseModel):
    index_name:str
    dir_id:str
    concurrency:int=2
