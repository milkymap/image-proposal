# text-semantic-embedding

# build image 

```bash
docker build -t image-proposal:gpu -f Dockerfile.gpu .
```

# run container 
```bash
    docker run --rm --name image-proposal -gpus all -v /path/to/transformers_cache/:/home/solver/transformers_cache -v /path/to/base_dir/:/home/solver/base_dir -e ES_HOST="elastic search hostname" -e ES_PORT=443 -e ES_SCHEME=https -e ES_BASIC_AUTH=XXXXXXX -p host_port:8000 image-proposal:gpu --port 8000 --host '0.0.0.0' --img_model_name clip-ViT-L-14   --nlp_model_name Sahajtomar/french_semantic --chunk_size 128 --nb_nlp_workers 1  --nb_img_workers 2 --mounting_path "/" 
```

