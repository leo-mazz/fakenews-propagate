import constants

from enum import Enum

import pandas as pd

StoryLabel = Enum('StoryLabel', 'FAKE REAL')

def get_news_from_csv(input_file):
    news = pd.read_csv(input_file, dtype=str)
    return news

def get_politifact_fake():
    return get_news_from_csv(constants.POLITIFACT_FAKE_PATH)

def get_politifact_real():
    return get_news_from_csv(constants.POLITIFACT_REAL_PATH)