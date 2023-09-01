from pydantic import BaseModel

class ImageRecommendation(BaseModel):
    text:str 
    nb_neighbors:int=16