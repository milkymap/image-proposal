import logging 

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(filename)s - %(lineno)3d >> %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(name='[image-proposal]')