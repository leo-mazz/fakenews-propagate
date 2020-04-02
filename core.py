import numpy as np
from numpy.linalg import norm

from news import StoryLabel

def normalized(a):
    return a / sum(a)

def propagate_credibility(init, weights, edges, news_audiences, threshold, use_domains, redistribution=0):
    s_users, s_news, s_domains = init
    w_users, w_domains = weights
    # (these are not just transposed objects: they have different
    # sparsity representation!)
    users_news, news_users, domains_news, news_domains = edges
    users_degrees = np.asarray(users_news.sum(axis=1)).ravel()
    news_degrees_users = np.asarray(news_users.sum(axis=1)).ravel()
    news_degrees_domains = np.asarray(news_domains.sum(axis=1)).ravel()
    domains_degrees = np.asarray(domains_news.sum(axis=1)).ravel()

    news_audiences = normalized(news_audiences)

    it = 0
    while True:
        # print(it, s_news)
        it += 1
        s_users = normalized(users_news.dot(s_news) / users_degrees)
        s_domains = normalized(domains_news.dot(s_news) / domains_degrees)

        prev_s_news = s_news
        s_news = w_users * (news_users.dot(s_users) / news_degrees_users)
        if use_domains:
            s_news += w_domains * (news_domains.dot(s_domains) / news_degrees_domains)
        s_news = ((redistribution * s_news.sum()) * news_audiences) + ((1-redistribution) * s_news)
        s_news = normalized(s_news)
        # print(it, norm(s_news - prev_s_news))
        if norm(s_news - prev_s_news) < threshold:
            print(f'Credibility propagation: returning at iteration {it}')
            return s_users, s_news, s_domains


def audience_credibility_rank(s_news, audiences, alpha):
    credibility_order = np.argsort(s_news)
    ranks = np.zeros_like(credibility_order)
    ranks[credibility_order] = np.arange(credibility_order.shape[0])

    middle = 0
    # audiences = normalized(audiences)
    audience_top_s_news = audiences[credibility_order][middle:]
    adjusted_ranks = np.zeros_like(ranks)
    adjusted_ranks[:middle] = ranks[:middle]
    adjusted_ranks[middle:] = ranks[middle:][np.argsort(audience_top_s_news)]

    return adjusted_ranks

def spared_audiences(true_labels, importance, audiences, budget, oracle=False):
    ordered_audiences = np.argsort(importance)[::-1]

    total_spared_audiences = 0
    fake_news_hit = 0
    i = -1
    while budget > 0 and i+1 < len(true_labels):
        i += 1
        story_index = ordered_audiences[i]
        lbl = true_labels[story_index]
        if lbl == StoryLabel.FAKE:
            budget -= 1
            fake_news_hit += 1
            total_spared_audiences += audiences[story_index]
        elif not oracle:
            budget -= 1
    
    return total_spared_audiences, fake_news_hit

def performance_of_targets(true_labels, s_news, news_audiences, news_users, budget):
    best_achievable, fake_news_hit = spared_audiences(true_labels, news_audiences, news_audiences, budget, oracle=True)
    print('Fake news found with oracle:', fake_news_hit)
    # rank = audience_credibility_rank(s_news, news_audiences)
    rank = s_news
    achieved, fake_news_hit = spared_audiences(true_labels, rank, news_audiences, budget)

    print('Fake news found given rank:', fake_news_hit)
    return float(achieved) / float(best_achievable) * 100