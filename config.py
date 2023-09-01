from enum import Enum 
from pydantic import BaseModel

class ZMQConfig(str, Enum):
    CLIENT2BROKER_ADDRESS:str="ipc:///tmp/client2broker.ipc"
    BROKER2WORKER_ADDRESS:str="inproc://broker2worker"
    WORKER2BROKER_ADDRESS:str="inproc://worker2broker"

class TaskStatus(str, Enum):
    FAILED:str="FAILED"
    UNDEFINED:str="UNDEFINED"

    PENDING:str="PENDING"
    RUNNING:str="RUNNING"

    TIMEDOUT:str="TIMEDOUT"
    COMPLETED:str="COMPLETED"
    INTERRUPTED:str="INTERRUPTED"

