import twitter_api
import constants

from collections import defaultdict
import pickle
import json

import pandas as pd
from tqdm import tqdm

class Tweet():
    def __init__(self, id, news, label):
        self.id = id
        self.news = news
        self.label = label
    
    def set_user(self, user):
        self.user = user
    
    def set_audience(self, audience):
        self.audience = audience

    
    def __str__(self):
        return f'{self.id},{self.news},{self.label},{self.user},{self.audience}'


def tweets_to_ids(tweets):
    return [t.id for t in tweets]

def tweets_to_news(tweets):
    return [t.news for t in tweets]

def tweets_to_labels(tweets):
    return [t.label for t in tweets]

def get_tweets_from_csv(input_file, label):
    all_tweets = []
    csv = pd.read_csv(input_file)
    for i in range(len(csv)):
        row = csv.iloc[i]
        news = row.id

        tweet_column = row.tweet_ids
        if not pd.isnull(tweet_column):
            tweets = row.tweet_ids.strip().split('\t')
            tweets = [Tweet(t_id, news, label) for t_id in tweets if t_id]
            all_tweets += tweets
    
    return all_tweets

def get_politifact_fake():
    return get_tweets_from_csv(constants.POLITIFACT_FAKE_PATH, 'fake')

def get_politifact_real():
    return get_tweets_from_csv(constants.POLITIFACT_REAL_PATH, 'real')



CATEGORY = 'statuses'
ENDPOINT = '/statuses/lookup'
def download_tweets(tweet_objects):
    budgets = twitter_api.make_budgets(900, 300, CATEGORY, ENDPOINT)
    t_interface = twitter_api.TwitterInterface(budgets)

    statuses = []
    audiences = {}

    for i in tqdm(range(0, len(tweet_objects), 100)):
        end_index = i + 100
        if len(tweet_objects) < end_index:
            end_index = len(tweet_objects)
        chunk = tweet_objects[i:i+100]

        ids = tweets_to_ids(chunk)
        news = tweets_to_news(chunk)
        labels = tweets_to_labels(chunk)

        id_news_map = dict(zip(ids, news))
        id_label_map = dict(zip(ids, labels))

        twitter = t_interface.get_connector()

        results = twitter.lookup_status(id=ids, include_entities=True)
        for r in results:
            if r:
                t_id = str(r['id'])
                screen_name = r['user']['screen_name']
                audience = r['user']['followers_count']

                status = Tweet(t_id, id_news_map[t_id], id_label_map[t_id])
                status.set_user(screen_name)
                status.set_audience(audience)

                statuses.append(status)
                audiences[screen_name] = audience
    
    return statuses, audiences

    




# all_users = defaultdict(set)
# ids = tweets_to_ids(ts)
# for i in range(0, 8900, 100):
#     chunk = ts[i:i+100]
#     ids = tweets_to_ids(chunk)

#     id_to_chunk = dict(zip(ids, chunk))

#     results = twitter.lookup_status(id=ids, include_entities=True, map=True)['id']
#     for t_id, value in results.items():
#         if value:
#             user = value['user']['id']
#             news = id_to_chunk[int(t_id)].news
#             all_users[user].add(news)


# with open('all_users_dump.pickle', 'wb') as output_file:
#     pickle.dump(all_users, output_file)

# with open('all_users_dump.pickle', 'rb') as input_file:
#     all_users_recovered = pickle.load(input_file)


# freq = defaultdict(int)
# for v in all_users_recovered.values():
#     freq[len(v)] += 1

# print(freq)