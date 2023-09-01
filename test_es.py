import asyncio 
from elasticsearch import AsyncElasticsearch

from uuid import uuid4 
import numpy as np 

import json
import requests

default_es_auth = 'Basic bWluaW5nOiZrVDNlMmRmN0RRRThidVd2LU5I'
default_es_uri = "https://q1mininges.ouest-france.fr"
default_es_index = "poc-image-proposal"
    
class ESConfig:
    def __init__(self, uri, auth, index):
        self.auth_header = {'Authorization': auth}
        self.uri = uri
        self.index = index

class IndexationMetier:
    
    def __init__(self, es_config):
        self.es_config = es_config
        
    def init_index_elasticsearch(self):
        print("### Load ES mapping file")
        es_mapping = {}
        """
        with open('mapping.json', 'r') as f: 
            es_mapping = json.load(f)
        print(es_mapping)
        """

        print("### Check ES liveness")
        es_headers = {}
        es_headers.update(self.es_config.auth_header)
        response = requests.get(self.es_config.uri, headers=es_headers, timeout=10)
        print(response.json())
        

        """
        print("### Check ES index")
        es_headers = {}
        es_headers.update(self.es_config.auth_header)
        response = requests.get(self.es_config.uri + "/" + self.es_config.index, headers=es_headers, timeout=5)
        print(response.json())
        
        if response.status_code != 200:
            print("### Init ES index")
            es_headers = {}
            es_headers.update(self.es_config.auth_header)
            es_headers.update({'content-type': 'application/json'})
            response = requests.put(self.es_config.uri + "/" + self.es_config.index, headers=es_headers, json=es_mapping, timeout=5)
            print(response.json())
        """

    def index_into_elascticsearch(self, data):
        es_headers = {}
        es_headers.update(self.es_config.auth_header)
        es_headers.update({'content-type': 'application/json'})
        response = requests.put(self.es_config.uri + "/" + self.es_config.index + "/_doc/" + data["id"], headers=es_headers, json=data, timeout=5)
        print(response.status_code)

########################################################
# MAIN
########################################################
def main(es_uri: str=default_es_uri, es_auth: str=default_es_auth, es_index: str=default_es_index):
    es_config = ESConfig(es_uri, es_auth, es_index)
    metier = IndexationMetier(es_config)
    metier.init_index_elasticsearch()

if __name__ == '__main__':
    main() 