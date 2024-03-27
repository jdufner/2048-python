import logging

logging.basicConfig(  # filename='../logs/puzzle.logs',
    encoding='utf-8',
    format='%(asctime)s,%(msecs)-3d - %(levelname)-8s - %(filename)s:%(lineno)d - '
           '%(module)s - %(funcName)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.DEBUG)
