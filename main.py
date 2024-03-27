from datetime import datetime
import logging
from rl.agent import train

if __name__ == '__main__':
    now: datetime = datetime.now()
    logging.basicConfig(filename=f'./logs/{now: %Y-%m-%d_%Hh%Mm%Ss}_main.log',
                        encoding='utf-8',
                        format='%(asctime)s,%(msecs)-3d - %(levelname)-8s - %(filename)s:%(lineno)d - '
                               '%(module)s - %(funcName)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)
    train()
