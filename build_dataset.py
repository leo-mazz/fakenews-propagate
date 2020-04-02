import tweets
from news import StoryLabel, get_politifact_fake,get_politifact_real
import constants

from urllib.parse import urlparse
import os
import random

import networkx as nx
import matplotlib.pyplot as plt
from scipy import sparse
import numpy as np

def create_directories_structure():
    def maybe_mkdir(path):
        if not os.path.exists(path):
            os.makedirs(path)
    
    maybe_mkdir(constants.DATA_ROOT)

def write_tweet_data(statuses, path):
    with open(path, 'w') as out_file:
        for s in statuses:
            out_file.write(str(s))
            out_file.write('\n')

def write_user_data(audiences, path):
    with open(path, 'w') as out_file:
        for u, a in audiences.items():
            out_file.write(f'{u},{a}')
            out_file.write('\n')

def extract_domain(url):
    if ('https://' not in url) and ('http://' not in url):
        url = 'http://' + url

    domain = urlparse(url).netloc

    if len(domain) >= 4 and domain[:4] == 'www.':
        domain = domain[4:]

    return domain

def make_graph(fake_news, real_news, statuses, audiences, min_user_occurrences):
    news = {} # news_id: (label, domain, users_set)
    domains = set()
    users = {} # screen_name: [audience, news_count]

    user_by_tid = {}
    for s in statuses:
        user_by_tid[s.id] = s.user

    def process_news(news_df, label):
        for i, row in news_df.iterrows():
            if not isinstance(row['news_url'], str) or not isinstance(row['tweet_ids'], str):
                continue

            dom = extract_domain(row['news_url'])
            domains.add(dom)
            id_and_label = row['id'] + str(label)
            news[id_and_label] = (label, dom, set())

            tweet_ids = row['tweet_ids'].strip().split('\t')
            for t_id in tweet_ids:
                if t_id in user_by_tid:
                    user = user_by_tid[t_id]

                    if user not in news[id_and_label][2]:
                        news[id_and_label][2].add(user)
                        if user in users:
                            users[user][1] += 1
                        else:
                            users[user] = [int(audiences[user]), 1]
    
    process_news(fake_news, StoryLabel.FAKE)
    process_news(real_news, StoryLabel.REAL)

    # MAKE DERIVATE GRAPHS
    # (drop users not sharing much, then stories with no user, then domains with no story)
    filtered_users = dict([(screen_name, data[0]) for screen_name, data in users.items() if data[1] >= min_user_occurrences])
    
    filtered_news = {}
    filtered_domains = set()
    for story_id, data in news.items():
        label, domain, sharers = data
        sharers = [u for u in sharers if u in filtered_users]
        if len(sharers) > 0:
            filtered_news[story_id] = (label, domain, sharers)
            filtered_domains.add(domain)
    
        
    # SHUFFLE ENTITIES!
    random.seed(304)
    shuffled_news = list(filtered_news.items())
    random.shuffle(shuffled_news)
    shuffled_users = list(filtered_users.items())
    random.shuffle(shuffled_users)
    shuffled_domains = list(filtered_domains)
    random.shuffle(shuffled_domains)

    # CREATE NODE DICTIONARIES
    user_nodes = dict([(data[0], (i, data[1])) for i, data in enumerate(shuffled_users)])
    story_nodes = dict([(items[0], (i, items[1][0])) for i, items in enumerate(shuffled_news)])
    domain_nodes = dict([(dom, (i, None)) for i, dom in enumerate(shuffled_domains)])

    # CREATE ADJACENCY MATRICES
    G = nx.Graph()

    id_to_index = lambda _id, dic: dic[_id][0]
    ids_to_indices = lambda ids, dic: [id_to_index(_id, dic) for _id in ids]

    users_news = np.zeros((len(user_nodes), len(story_nodes)), dtype=np.int)
    domains_news = np.zeros((len(domain_nodes), len(story_nodes)), dtype=np.int)

    for story_id, data in shuffled_news:
        label, domain, sharers = data
        story_index = id_to_index(story_id, story_nodes)
        users_news[ids_to_indices(sharers, user_nodes), story_index] = 1
        domains_news[id_to_index(domain, domain_nodes), story_index] = 1

        # condition = (label == StoryLabel.FAKE)
        # condition = (label == StoryLabel.REAL)
        condition = True
        if condition:
            G.add_node(story_id)
            # G.add_node(domain)
            # G.add_edge(story_id, domain)
            for u in sharers:
                G.add_node(u)
                G.add_edge(story_id, u)

    # CREATE SPARSE ADJACENCY MATRICES
    news_users = users_news.transpose()
    users_news = sparse.csr_matrix(users_news)
    news_users = sparse.csr_matrix(news_users)

    news_domains = domains_news.T
    domains_news = sparse.csr_matrix(domains_news)
    news_domains = sparse.csr_matrix(news_domains)

    nodes = user_nodes, story_nodes, domain_nodes
    edges = users_news, news_users, domains_news, news_domains

    return nodes, edges, G


def fetch_from_twitter():
    # GATHER LABELLED TWEETS
    fake_tweets = tweets.get_politifact_fake()
    real_tweets = tweets.get_politifact_real()

    # CREATE DIRECTORY STRUCTURE
    create_directories_structure()

    # FETCH FROM TWITTER
    all_tweets = fake_tweets + real_tweets

    statuses, audiences = tweets.download_tweets(all_tweets)

    # WRITE TO DISK
    write_tweet_data(statuses, constants.STATUSES)
    write_user_data(audiences, constants.USERS)

def read_twitter_data():
    statuses = []
    audiences = {}
    with open(constants.STATUSES, 'r') as s_file:
        for line in s_file:
            values = line.strip().split(',')
            if len(values) == 5:
                status = tweets.Tweet(values[0], values[1], values[2])
                status.set_user(values[3])
                status.set_audience(values[4])
                statuses.append(status)
    
    with open(constants.USERS, 'r') as u_file:
        for line in u_file:
            if line:
                values = line.strip().split(',')
                if len(values) == 2:
                    audiences[values[0]] = values[1]

    return statuses, audiences
        
def get_me_a_graph(min_user_occurrences):
    statuses, audiences = read_twitter_data()

    fake_news = get_politifact_fake()
    real_news = get_politifact_real()
    nodes, edges, G = make_graph(fake_news, real_news, statuses, audiences, min_user_occurrences)

    return nodes, edges, G


def main():
    # Example usage: 

    # SAVE ON DISK TWEET AND USER DATA
    # fetch_from_twitter()

    # READ FROM DISK TWEET AND USER DATA
    # statuses, audiences = read_twitter_data()

    # fake_news = get_politifact_fake()
    # real_news = get_politifact_real()
    # nodes, edges, G = make_graph(fake_news, real_news, statuses, audiences, 4)

    # DO STUFF WITH NETWORK, E.G. DISPLAY IT
    # print('preparing to show network')
    # plt.subplot(121)
    # nx.draw(G, node_size=20)
    # plt.show()
    pass


if __name__ == '__main__':
    main()