import zmq 
import pickle 

import click 
from log import logger 
from config import ZMQConfig

from os import path 
from glob import glob 


@click.command()
@click.option('--path2corpus', help='', type=click.Path(exists=True, file_okay=False))
@click.option('--extension', type=click.Choice(['*.txt', '*.jpg']))
def main(path2corpus:str, extension:str) -> None:
    filepaths = glob(path.join(path2corpus, extension))
    logger.info(f'nb files {len(filepaths)}')

    ctx = zmq.Context()
    dealer_socket = ctx.socket(zmq.DEALER)
    dealer_socket.connect(ZMQConfig.CLIENT2BROKER_ADDRESS)

    try:
        for path2file in filepaths:
            if extension == '*.txt':
                with open(path2file, mode='r') as fp:
                    text = fp.read()
                    binarystream = text.encode()
                print(text[:100])
                topic = b'TEXT'
            elif extension == '*.jpg':
                with open(path2file, mode='rb') as fp:
                    binarystream = fp.read()

                print(binarystream[:10])
                topic = b'IMAGE'
            else:
                break 

            dealer_socket.send_multipart([
                b'',
                topic,
                binarystream
            ])
            _, response = dealer_socket.recv_multipart()
            print(pickle.loads(response))

    except KeyboardInterrupt:
        pass 
    except Exception as e:
        logger.error(e)
    finally:
        dealer_socket.close(linger=0)
        ctx.term()



if __name__ == '__main__':
    main()
