from news import StoryLabel

import numpy as np

def average_precision(scores_and_labels):
    precisions = []
    recalls = []
    
    for middle_point in range(1, len(scores_and_labels)-1):
        reals = scores_and_labels[:middle_point]
        fakes = scores_and_labels[middle_point:]

        count_label = lambda lst, lbl: sum([1 for actual, predicted in lst if predicted == lbl])

        true_positives = count_label(fakes, StoryLabel.FAKE)
        precisions.append(true_positives / len(fakes))
        recalls.append(true_positives / (count_label(scores_and_labels, StoryLabel.FAKE)))
    
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    recall_left_shift = np.concatenate([np.ones(1), recalls[:-1]])
    recall_deltas = recall_left_shift - recalls

    return recall_deltas.dot(precisions)

def evaluate_middle_split(scores_and_labels):
    middle_point = len(scores_and_labels) // 2
    reals = scores_and_labels[:middle_point]
    fakes = scores_and_labels[middle_point:]

    count_label = lambda lst, lbl: sum([1 for actual, predicted in lst if predicted == lbl])

    true_positives = count_label(fakes, StoryLabel.FAKE)
    precision = true_positives / len(fakes)
    recall = true_positives / (count_label(scores_and_labels, StoryLabel.FAKE))

    f1 = 2 * (precision * recall) / (precision + recall)

    true_negatives = count_label(reals, StoryLabel.REAL)
    accuracy = (true_positives + true_negatives) / len(scores_and_labels)
    return f1, accuracy

def initialize_scores_rnd(true_labels_values, default_label, percentage_labelled, seed):
    np.random.seed(seed)
    n_labelled = int(percentage_labelled * len(true_labels_values))
    random_indices = np.random.choice(np.arange(len(true_labels_values)), size=n_labelled, replace=False)

    init = np.zeros_like(true_labels_values)
    init[:] = default_label
    
    init[random_indices] = true_labels_values[random_indices]

    # print(true_labels_values[random_indices])

    return init

def initialize_scores_deg(true_labels_values, default_label, percentage_labelled, news_users):
    n_labelled = int(percentage_labelled * len(true_labels_values))
    news_degrees = np.asarray(news_users.sum(axis=1)).ravel()
    highest_degs = np.argsort(news_degrees)[::-1]

    init = np.zeros_like(true_labels_values)
    init[:] = default_label
    
    init[highest_degs[:n_labelled]] = true_labels_values[highest_degs[:n_labelled]]

    # print(true_labels_values[highest_degs[:n_labelled]])

    return init

def initialize_scores_balanced_deg(true_labels_values, default_label, percentage_labelled, news_users):
    n_labelled = int(percentage_labelled * len(true_labels_values))
    news_degrees = np.asarray(news_users.sum(axis=1)).ravel()
    highest_degs = np.argsort(news_degrees)[::-1]

    count_fake_init = 0
    count_real_init = 0
    balanced_highest_degs = []
    for i, j in enumerate(highest_degs):
        if count_fake_init + count_real_init >= n_labelled:
            break
        if int(true_labels_values[j]) == 1:
            if count_fake_init < n_labelled // 2:
                count_fake_init += 1
            else:
                continue
        if int(true_labels_values[j]) == 0:
            print('detected real story index')
            if count_real_init < n_labelled - (n_labelled // 2):
                count_real_init += 1
            else:
                continue

        print('reaching end')
        balanced_highest_degs.append(j)

    init = np.zeros_like(true_labels_values)
    init[:] = default_label
    
    init[balanced_highest_degs] = true_labels_values[balanced_highest_degs]

    # print(true_labels_values[balanced_highest_degs])

    return init