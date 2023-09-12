from enum import Enum 

class ZMQConfig(str, Enum):
    CLIENT2BROKER_ADDRESS:str="tcp://*:1200"  #"ipc:///tmp/client2broker.ipc"
    BROKER2WORKER_ADDRESS:str="inproc://broker2worker"
    WORKER2BROKER_ADDRESS:str="inproc://worker2broker"


