import logging
import sys

logger = logging.getLogger()
formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')

# Stream to file
# stream = logging.StreamHandler()
# stream.setFormatter(formatter)
# logger.addHandler(stream)

# stdout
stdout = logging.StreamHandler(sys.stdout)
stdout.setLevel(logging.DEBUG)
stdout.setFormatter(formatter)
logger.addHandler(stdout)

logger.setLevel(logging.DEBUG)
