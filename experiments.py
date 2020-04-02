from build_dataset import get_me_a_graph
from core import propagate_credibility, performance_of_targets
from news import StoryLabel
from experiment_utils import evaluate_middle_split, initialize_scores_rnd, initialize_scores_deg, initialize_scores_balanced_deg, average_precision

import functools
import operator

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def describe_graph(nodes, edges, G):
    # DESCRIBE COMPOSITION OF NEWS
    user_nodes, story_nodes, domain_nodes = nodes
    users_news, news_users, domains_news, news_domains = edges
    only_label = lambda lbl, news: dict([(k, v) for k, v in news.items() if v[1] == lbl])
    fake_news_nodes = only_label(StoryLabel.FAKE, story_nodes)
    real_news_nodes = only_label(StoryLabel.REAL, story_nodes)
    print(f'There is a total of {len(story_nodes)} stories: {len(fake_news_nodes)} fake, {len(real_news_nodes)} real')

    # DESCRIBE COMPOSITION OF USERS
    user_lists = lambda news: [list(np.where(news_users[v[0]].toarray().ravel())[0]) for v in news.values()]
    user_lists_fake = user_lists(fake_news_nodes)
    user_lists_real = user_lists(real_news_nodes)
    set_users_fake = set(functools.reduce(operator.iconcat, user_lists_fake, []))
    set_users_real = set(functools.reduce(operator.iconcat, user_lists_real, []))
    set_users_real_and_fake = set_users_fake & set_users_real
    print(f'There is a total of {len(user_nodes)} users: {len(set_users_fake)} have shared at least 1 fake story, {len(set_users_real)} have shared at least 1 real story, {len(set_users_real_and_fake)} have shared one of each')

    # DESCRIBE COMPOSITION OF DOMAINS
    domain_lists = lambda news: [list(np.where(news_domains[v[0]].toarray().ravel())[0]) for v in news.values()]
    domain_lists_fake = domain_lists(fake_news_nodes)
    domain_lists_real = domain_lists(real_news_nodes)
    set_domains_fake = set(functools.reduce(operator.iconcat, domain_lists_fake, []))
    set_domains_real = set(functools.reduce(operator.iconcat, domain_lists_real, []))
    set_domains_real_and_fake = set_domains_fake & set_domains_real
    print(f'There is a total of {len(domain_nodes)} domains: {len(set_domains_fake)} have shared at least 1 fake story, {len(set_domains_real)} have shared at least 1 real story, {len(set_domains_real_and_fake)} have shared one of each')
    
    cross_domain_counts = 0
    for lst in domain_lists(story_nodes):
        ambiguous_domains = False
        for d in lst:
            if d in set_domains_real_and_fake:
                ambiguous_domains = True
        if ambiguous_domains:
            cross_domain_counts += 1

    print(f'News with ambiguous domains: {cross_domain_counts}')


    cross_user_counts = 0
    for lst in user_lists(story_nodes):
        ambiguous_users = False
        for d in lst:
            if d in set_users_real_and_fake:
                ambiguous_users = True
        if ambiguous_users:
            cross_user_counts += 1

    print(f'News with ambiguous users: {cross_user_counts}')

    # INSPECT CONNECTEDNESS PROPERTIES
    # (this depends on how G is constructed in build_dataset.py, e.g. you can exclude domains, or only look at one kind of label)
    print(f'Connected graph? {nx.is_connected(G)}')
    # (all story nodes have the prefix 'StoryLabel' in their name. You can filter by label, using e.g. 'StoryLabel.FAKE')
    print([len([1 for n in c if 'StoryLabel' in n]) for c in sorted(nx.connected_components(G), key=len, reverse=True)])


def basic_propagation(nodes, edges, news_audiences, percentage_init, rnd_init, use_domains, redistribution):
    user_nodes, story_nodes, domain_nodes = nodes
    users_news, news_users, domains_news, news_domains = edges

    story_tuples = sorted(list(story_nodes.values())) # sort stories by int index

    labels_values = np.array([(1.0 if lbl == StoryLabel.FAKE else 0.0) for _, lbl in story_tuples])
    if rnd_init:
        stories_init = initialize_scores_rnd(labels_values, 0.5, percentage_init, rnd_init)
    else:
        stories_init = initialize_scores_deg(labels_values, 0.5, percentage_init, news_users)
    # print(stories_init)

    s_users, s_news, s_domains = propagate_credibility([None, stories_init, None], (0.5, 0.5), edges, news_audiences, 0.00000001, use_domains=use_domains, redistribution=redistribution)

    # print('f1score, accuracy before propagation', evaluate_middle_split(story_tuples))
    
    true_labels = [lbl for _, lbl in story_tuples]
    news_scores_and_labels = sorted(list(zip(s_news, true_labels)), key=lambda tup: tup[0])
    print('f1score, accuracy after propagation', evaluate_middle_split(news_scores_and_labels))
    print('avg precision', average_precision(news_scores_and_labels))

    scores = (s_users, s_news, s_domains)

    return scores, average_precision(news_scores_and_labels)

    # labels_per_user = {}
    # for i, lbl in story_tuples:
    #     num_lbl = 1 if lbl == StoryLabel.FAKE else 0
    #     rel_users = list(np.arange(news_users.shape[1])[news_users[i].toarray().ravel() == 1])
    #     for u in rel_users:
    #         if u not in labels_per_user:
    #             labels_per_user[u] = [0,0]
    #         labels_per_user[u][num_lbl] += 1
    
    # labels_per_user = sorted(list(labels_per_user.items()))
    # labels_per_user = [counts for i, counts in labels_per_user]
    # users_scores_and_labels = sorted(list(zip(s_users, labels_per_user)), key=lambda tup: tup[0])
    # print(users_scores_and_labels)

    # labels_per_domain = {}
    # for i, lbl in story_tuples:
    #     num_lbl = 1 if lbl == StoryLabel.FAKE else 0
    #     rel_domains = list(np.arange(news_domains.shape[1])[news_domains[i].toarray().ravel() == 1])
    #     for d in rel_domains:
    #         if d not in labels_per_domain:
    #             labels_per_domain[d] = [0,0]
    #         labels_per_domain[d][num_lbl] += 1

    # labels_per_domain = sorted(list(labels_per_domain.items()))
    # labels_per_domain = [counts for i, counts in labels_per_domain]
    # domains_scores_and_labels = sorted(list(zip(s_domains, labels_per_domain)), key=lambda tup: tup[0])
    # print(domains_scores_and_labels)

def inspect_ranks(true_labels, s_news, news_users, news_audiences):
    # (the following code shows that most of the audience mass is in misclassified fake news)
    by_credibility = sorted(list(zip(s_news, list(zip(news_audiences, true_labels)))), key=lambda tup: tup[0])
    audiences_labels = [b for a, b in by_credibility]
    by_audiences = sorted(list(zip(audiences_labels, list(range(len(audiences_labels))))), key=lambda tup: tup[0][0])
    false_negatives = [(a[0], a[1]) for a, b in by_audiences if a[1] == StoryLabel.FAKE and b<=(len(by_audiences)//2)]
    false_negative_audiences = [a for a, b in false_negatives]
    print(f'Audience mass for false negatives: {sum(false_negative_audiences)}')
    print(len(false_negative_audiences))

    true_positives = [(a[0], a[1]) for a, b in by_audiences if a[1] == StoryLabel.FAKE and b>(len(by_audiences)//2)]
    true_positives_audiences = [a for a, b in true_positives]
    print(f'Audience mass for true positives: {sum(true_positives_audiences)}')
    print(len(true_positives_audiences))

    # PLOT SCORE AGAINST AUDIENCE SIZE, MARKING MIDDLE SPLIT POINT
    yy = [b[0] for a, b in by_credibility if b[1] == StoryLabel.FAKE]
    xx = [a for a, b in by_credibility if b[1] == StoryLabel.FAKE]
    plt.plot(xx, yy)
    plt.scatter([sorted(s_news)[len(true_labels) // 2]], [0], c='red')
    plt.ylabel('audiences')
    plt.xlabel('score')
    plt.show()



def show_graph(G):
    # UNLESS THE GRAPH IS SMALL, THIS WILL HANG
    plt.subplot(121)
    nx.draw(G, node_size=20)
    plt.show()

def main():
    # CONSTRUCT GRAPH
    nodes, edges, G = get_me_a_graph(4)
    user_nodes, story_nodes, _ = nodes
    _, news_users, _, _ = edges

    # DESCRIBE BASIC FEATURES OF GRAPH
    # describe_graph(nodes, edges, G)

    # DISPLAY GRAPH (NOT RECCOMENDED, GRAPHS ARE HUGE)
    # show_graph(G) # It's reccomended not to run this

    # COMPUTE PRECISION STATS WITH RANDOM INITS
    # precs = []
    # for seed in range(50):
    #     _, prec = basic_propagation(nodes, edges, 1.0/100, rnd_init=seed, use_domains=True, redistribution=0)
    #     precs.append(prec)
    
    # precs = np.array(precs)
    # print(f'avg across runs was {precs.sum() / len(precs)}')
    # print(f'std across runs was {precs.std()}')
    # print(f'min across runs was {np.min(precs)}')
    # print(f'max across runs was {np.max(precs)}')

    # ------ CODE TO COMPUTE NEWS AUDIENCES ---------
    story_tuples = sorted(list(story_nodes.values())) # sort stories by int index
    true_labels = np.array([lbl for _, lbl in story_tuples])
    user_tuples = sorted(list(user_nodes.values())) # sort users by int index
    user_audiences = np.array([aud for _, aud in user_tuples])
    user_lists = [list(np.where(news_users[i].toarray().ravel())[0]) for i in range(len(true_labels))]
    news_audiences = [sum([user_audiences[u] for u in lst]) for lst in user_lists]
        
    scores, _ = basic_propagation(nodes, edges, np.array(news_audiences), 1.0/100, rnd_init=None, use_domains=False, redistribution=0.1)
    s_users, s_news, s_domains = scores
    # ----------------- END -----------------------

    # DISPLAY DISTRIBUTION OF AUDIENCE MASS FOR FAKE NEWS
    # inspect_ranks(true_labels, s_news, news_users, news_audiences)

    # COMPUTE SPARED AUDIENCE RATIO
    r = performance_of_targets(true_labels, s_news, np.array(news_audiences), news_users, budget=24)
    print(f'Spared audience ratio:  {r}')

if __name__ == '__main__':
    main()