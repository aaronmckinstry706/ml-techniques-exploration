"""
Contains the absolute paths of the source and data directories. 
"""

import logging
import os
import os.path as path

logging.basicConfig()
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

SOURCE_DIRECTORY = path.abspath(__file__)
DATA_DIRECTORY = path.abspath(SOURCE_DIRECTORY + "/../data")
TITANIC_DATA_DIRECTORY = path.abspath(DATA_DIRECTORY + "/titanic")

if __name__ == '__main__':
    LOGGER.info("SOURCE_DIRECTORY = " + SOURCE_DIRECTORY)
    LOGGER.info("DATA_DIRECTORY = " + DATA_DIRECTORY)
    LOGGER.info("TITANIC_DATA_DIRECTORY = " + TITANIC_DATA_DIRECTORY)