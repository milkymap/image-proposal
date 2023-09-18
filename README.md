# text-semantic-embedding

# build image 

```bash
docker build -t image-proposal:gpu -f Dockerfile.gpu .
```

# run container 
```bash
docker run -it --rm --gpus all --name image-proposal --network proposal  -v /path/to/transformers_cache/:/home/solver/transformers_cache -v /path/to/base_dir/:/home/solver/base_dir -e ES_HOST=elasticsearch-server -e ES_PORT=9200 -e ES_SCHEME=http -e ES_BASIC_AUTH='&jmpA65Z90' -p 8000:8000 -e NB_SHARDS_FOR_KNOWLEDGE_INDEX=2 -e NB_SHARDS_FOR_IMAGE_INDEX=1 image-proposal:gpu --port 8000 --host '0.0.0.0' --img_model_name clip-ViT-L-14 --nlp_model_name Sahajtomar/french_semantic --chunk_size 128 --nb_nlp_workers 1  --nb_img_workers 1 --mounting_path "/"  --knowledge_index_name "knowledge_index" --image_index_name "image_index"
```


