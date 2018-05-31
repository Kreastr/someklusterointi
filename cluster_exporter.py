# -*- coding: utf-8 -*-
from datetime import datetime
from datetime import timedelta
import math
import simplejson
import numpy as np
from sklearn.manifold import TSNE
import pickle

from vecs import Vecs

from generalClassifierInterface import generalClassifierInterface
from pos_filter import pos_filter
from sentiment import getSentiment
import clustering

ru_vocab, ru_vecs = 'vocab.txt', 'vecs.bin'

print('Using vecs "%s" and vocab "%s" for russian tags' % (ru_vecs, ru_vocab))
ru_tag_classifier = generalClassifierInterface(Vecs(ru_vocab, ru_vecs), 'yandex_topic.clf', 'yandex_topic_tags.dict')

print('Using vecs "s24_swivel_pickle.bin" for finnish tags')
with open('s24_swivel_pickle.bin') as f:
    fi_vecs = pickle.load(f)
fi_tag_classifier = generalClassifierInterface(fi_vecs, 'yle_topic.clf', 'yle_topic_tags.dict')

tag_label_overrides = {u'auto_racing': u'Auto Racing', u'index': u'Breaking News', u'martial_arts': u'Martial Arts'}


# if the maximum or average is above any one of these thresholds, the cluster is saved
GROWTH_RATE_MAX_THRESHOLD = 5
GROWTH_RATE_AVG_THRESHOLD = 1

def filter_interesting_clusters(clusters):
    # all clusters that have a minimum x tweets/h growth rate at some point,
    # and have high enough entropy are deemed "interesting".
    interesting_clusters = []
    for c in clusters:
        modifier = 1
        if c.lang == 'fi':
            modifier = 0.3

        if len(c.hourly_growth_rate) == 0:
            continue

        if (any([g > GROWTH_RATE_MAX_THRESHOLD * modifier for g in c.hourly_growth_rate]) \
            or sum(c.hourly_growth_rate) / len(c.hourly_growth_rate) > GROWTH_RATE_AVG_THRESHOLD * modifier)\
           and calculate_cluster_entropy(c) > clustering.ENTROPY_THRESHOLD:
            interesting_clusters.append(c)

    return interesting_clusters


# TODO optimize by collecting all clusters' tweets at the same time to not need to
#      read through the original data files several times.
def save_cluster_texts(clusters_to_save):
    for i in range(len(clusters_to_save)):
        c = clusters_to_save[i]

        print('Cluster %d/%d, id: %d' % (i + 1, len(clusters_to_save), c.id))

        text_list = []
        tweet_timestamps = {}

        # fetch the original tweet text
        with open('cluster_data/cluster_%d.json' % c.id, 'w') as f:
            for t in c.text:
                parts = t.split(' ')
                tweet_timestamps[int(parts[1])] = int(parts[0])

            if c.lang == 'ru':
                # the cluster's tweets can only be in two different days' data
                for i in range(2):
                    tweet_date = datetime.utcfromtimestamp(c.created_at / 1000) + timedelta(days=i)
                    with open('tweets_%d/%02d/%02d.ru.json' % (tweet_date.year, tweet_date.month, tweet_date.day)) as f_tweets:
                        for l in f_tweets:
                            obj = simplejson.loads(l)
                            id = int(obj['id'])
                            if id in tweet_timestamps:
                                tweet_object = {
                                                'text': obj['text'],
                                                'screen_name': obj['user']['screen_name'],
                                                'id': str(obj['id']),
                                                't': tweet_timestamps[id]
                                               }

                                if 'geo' in obj and (obj['geo'] is not None):
                                    tweet_object['geo'] = obj['geo']

                                text_list.append(tweet_object)
            elif c.lang == 'fi':
                with open('/home/kosomaa/fi_tweets_turku_scraped/fi_mh-17.json') as f_tweets:
                    tweets = simplejson.load(f_tweets)
                    for t in tweets:
                        if int(t['id']) in ids:
                            text_list.append({'text': t['text'],
                                'screen_name': t['user'],
                                'id': t['id']})



            text_list.sort(key=lambda tweet: tweet['t'])
            simplejson.dump(text_list, f)

# returns a dictionary ready to be saved as a json file
def convert_to_dict(clusters_to_filter, ru_idfs, fi_idfs):
    if isinstance(clusters_to_filter, dict):
        clusters_to_filter = clusters_to_filter.values()

    clusters_to_save = filter_interesting_clusters(clusters_to_filter)
    json_formatted = []

    cdata = [c.center / c.norm for c in clusters_to_save]
    if len(cdata) < 5:
        return json_formatted
        
    t_sne_space = TSNE(n_components=2, metric='cosine').fit_transform(cdata)
    # normalize T-SNE space to -1 to 1
    minimums = t_sne_space.min(axis=0)
    maximums = t_sne_space.max(axis=0)
    for v in t_sne_space:
        v[0] = 2 * (v[0] - minimums[0]) / (maximums[0] - minimums[0]) - 1
        v[1] = 2 * (v[1] - minimums[1]) / (maximums[1] - minimums[1]) - 1

    for cluster_index in range(len(clusters_to_save)):
        c = clusters_to_save[cluster_index]

        idfs = ru_idfs if c.lang == 'ru' else fi_idfs

        # TODO remove temporary filtering
        if (c.created_at < 1405555200000): # 17/07/2014 00:00:00
            continue
        #if (c.created_at < 1503014400000): # 18/08/2017 00:00:00
            #continue

        if len(c.hourly_growth_rate) < 1:
            continue

        for i in range(len(c.hourly_growth_rate) + 1):
            update = {}
            # timestamp
            update['t'] = int(c.first_growth_time / 1000) + i * 60 * 60

            # start with a new cluster event
            if i == 0:

                words = []
                for t in c.text[:c.last_size]:
                    for w in t.replace('#', '').split(' '):
                        if w != '':
                            words.append(w)

                total_sentiment = c.hourly_accum_sentiment[len(c.hourly_accum_sentiment) - 1]

                if c.lang == 'ru':
                    tags = ru_tag_classifier.getTags([words], top_n=3)
                elif c.lang == 'fi':
                    tags = fi_tag_classifier.getTags([words], top_n=3)

                tags = [tag_label_overrides.get(t, t.title()) for t in tags]


                #get_keywords(c, idfs)[:4],
                update['n'] = {c.id:                                                \
                                {                                                   \
                                  's': round(c.hourly_growth_rate[i]),              \
                                  'k': c.hourly_keywords[i],                        \
                                  'lang': c.lang,                                   \
                                  'sentiment': round(c.hourly_sentiment[i], 3),     \
                                  'sentiment_total': round(total_sentiment, 3),     \
                                  'tags': tags if tags is not None else [],         \
                                  't_sne': [float(t_sne_space[cluster_index][0]),   \
                                            float(t_sne_space[cluster_index][1])]   \
                                }                                                   \
                              }
            elif i == len(c.hourly_growth_rate):
                # insert a negative number at the end to mark the end of the cluster
                update['u'] = {c.id: {'s': -1}}
            else:
                update['u'] = {c.id: {'s': int(round(c.hourly_growth_rate[i])), 'sentiment': round(c.hourly_sentiment[i], 3), 'sentiment_accum': round(c.hourly_accum_sentiment[i], 3), 'k': c.hourly_keywords[i]}}

            json_formatted.append(update)

    json_formatted.sort(key=lambda update: update['t'])
    return json_formatted


def get_keywords(cluster, idfs):
    word_freqs = {}
    for t in cluster.text[:cluster.last_size]:
        for w in t.split(' ')[2:]:
            if w != '':
                if w in word_freqs:
                    word_freqs[w] += 1
                else:
                    word_freqs[w] = 0

    for w, c in word_freqs.iteritems():
        # filter based on part of speech
        if pos_filter(w, ['A', 'ADV', 'S', 'V']) and w != 'быть':
            if w in idfs:
                word_freqs[w] *= idfs[w]
        else:
            word_freqs[w] = 0

    return sorted(word_freqs, key=word_freqs.get, reverse=True)

def calculate_cluster_entropy(cluster):
    word_counts = {}
    total_word_count = 0

    for t in cluster.text:
        for w in t.split(' ')[2:]:
            if w == '':
                continue

            if w in word_counts:
                word_counts[w] += 1
            else:
                word_counts[w] = 1

            total_word_count += 1

    entropy = 0
    for w, c in word_counts.iteritems():
        prob = float(c) / total_word_count
        entropy += -prob * math.log(prob)

    return entropy


