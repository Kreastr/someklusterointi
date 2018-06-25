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
        
        # filter excessively small clusters
        if len(c.documents) < 20:
            continue

        if (any([g > GROWTH_RATE_MAX_THRESHOLD * modifier for g in c.hourly_growth_rate]) \
            or sum(c.hourly_growth_rate) / len(c.hourly_growth_rate) > GROWTH_RATE_AVG_THRESHOLD * modifier):
            interesting_clusters.append(c)
#\
#           and calculate_cluster_entropy(c) > clustering.ENTROPY_THRESHOLD
    return interesting_clusters


def collectDataForCluster(c, cache, end_time):
    chash = hash(simplejson.dumps(list(map(lambda x: x[0], c.text_data))))#+str(end_time));
    if chash in cache:
        return cache[chash]
    
    text_list = []
    tweet_timestamps = {}
    end_time -= 3*3600*1000
    for t in c.text_data:
        parts = t[0][0].split(' ')
        #if (end_time+3*3600*1000) < int(parts[0]):
        #    continue
        #if (end_time-3*3600*1000) > (int(parts[0])):
        #    continue
            
        tweet_timestamps[int(parts[1])] = int(parts[0])
        text_list.append({'text': t[0][1],
                        'screen_name': t[0][2],
                        't': int(parts[0]),#t['user'],
                        'id': int(parts[1])})#t['id']})

    #if c.lang == 'ru':
        ## the cluster's tweets can only be in two different days' data
        #for i in range(2):
            #tweet_date = datetime.utcfromtimestamp(c.created_at / 1000) + timedelta(days=i)
            #with open('tweets_%d/%02d/%02d.ru.json' % (tweet_date.year, tweet_date.month, tweet_date.day)) as f_tweets:
                #for l in f_tweets:
                    #obj = simplejson.loads(l)
                    #id = int(obj['id'])
                    #if id in tweet_timestamps:
                        #tweet_object = {
                                        #'text': obj['text'],
                                        #'screen_name': obj['user']['screen_name'],
                                        #'id': str(obj['id']),
                                        #'t': tweet_timestamps[id]
                                       #}

                        #if 'geo' in obj and (obj['geo'] is not None):
                            #tweet_object['geo'] = obj['geo']

                        #text_list.append(tweet_object)
    #elif c.lang == 'fi':
        #with open('/home/kosomaa/fi_tweets_turku_scraped/fi_mh-17.json') as f_tweets:
            #tweets = simplejson.load(f_tweets)
            #for t in tweets:
                #if int(t['id']) in ids:
                    #text_list.append({'text': t['text'],
                        #'screen_name': t['user'],
                        #'id': t['id']})



    text_list.sort(key=lambda tweet: tweet['t'])
    cache[chash] = simplejson.dumps(text_list)
    return cache[chash]

# TODO optimize by collecting all clusters' tweets at the same time to not need to
#      read through the original data files several times.
def save_cluster_texts(clusters_to_save):
    for i in range(len(clusters_to_save)):
        c = clusters_to_save[i]

        print('Cluster %d/%d, id: %d' % (i + 1, len(clusters_to_save), c.id))

        # fetch the original tweet text
        with open('cluster_data/cluster_%d.json' % c.id, 'w') as f:
            
            text_list_str = collectDataForCluster(c)
            f.write = text_list_str

def getTagsForTexts(c, tweets_accum):
    words = []
    for t in tweets_accum:
        for w in t.replace('#', '').split(' '):
            if w != '':
                words.append(w)
    if c.lang == 'ru':
        tags = ru_tag_classifier.getTags([words], top_n=3)
    elif c.lang == 'fi':
        tags = fi_tag_classifier.getTags([words], top_n=3)
    return tags

# returns a dictionary ready to be saved as a json file
def convert_to_dict(clusters_to_filter, ru_idfs, fi_idfs, start_time):
    print (start_time)
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
        # 
        if (c.created_at < (start_time-len(c.hourly_growth_rate)*3600*1000)):#1405555200000): # 17/07/2014 00:00:00
            continue
        #if (c.created_at < 1503014400000): # 18/08/2017 00:00:00
            #continue

        if len(c.hourly_growth_rate) < 1:
            continue


        start_idx = max(int((c.created_at-start_time)/3600/1000),1)
        for i in range(start_idx, len(c.hourly_growth_rate)):
            update = {}
            # timestamp
            update['t'] = int(c.first_growth_time / 1000) + i * 60 * 60

            # start with a new cluster event
            if i == start_idx:

                total_sentiment = c.hourly_accum_sentiment[len(c.hourly_accum_sentiment) - 1]
                
                tags = c.hourly_tags[len(c.hourly_tags) - 1]
                    
                if tags is not None:
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
                update['u'] = {c.id: {'s': int(round(c.hourly_growth_rate[i])), 'sentiment': round(c.hourly_sentiment[i], 3), 'sentiment_accum': round(c.hourly_accum_sentiment[i], 3), 'k': c.hourly_keywords[i-1]}}

            json_formatted.append(update)

    json_formatted.sort(key=lambda update: update['t'])
    return json_formatted


def get_keywords_for_message_list(text_data, idfs):
    word_freqs = {}
    for t in text_data:
        #print (t)
        #print ("\n".join(t.split(' ')[2:]))
        for w in t.split(' ')[2:]:
            if w != '':
                if w in word_freqs:
                    word_freqs[w] += 1
                else:
                    word_freqs[w] = 1
    #print (word_freqs)
    for w, c in word_freqs.iteritems():
        # filter based on part of speech
        try:
            if pos_filter(w, ['A', 'ADV', 'S', 'V']) and (not w in ['http','быть']):
                if w in idfs:
                    word_freqs[w] *= idfs[w]
            else:
                word_freqs[w] = 0
        except:
            word_freqs[w] = 0
            
    return sorted(word_freqs, key=word_freqs.get, reverse=True)

def get_keywords(cluster, idfs):
    word_freqs = {}
    for t in cluster.text_data:
        for w in t[0][0].split(' ')[2:]:
            #print (w)
            if w != '':
                if w in word_freqs:
                    word_freqs[w] += 1
                    #print ('++')
                else:
                    word_freqs[w] = 1
                    #print ('=1')
    #print (word_freqs)
    for w, c in word_freqs.iteritems():
        # filter based on part of speech
        try:
            if pos_filter(w, ['A', 'ADV', 'S', 'V']) and (not w in ['http','быть']):
                if w in idfs:
                    word_freqs[w] *= idfs[w]
            else:
                word_freqs[w] = 0
        except:
            word_freqs[w] = 0
            
    return sorted(word_freqs, key=word_freqs.get, reverse=True)

def calculate_cluster_entropy(cluster, idfs):
    word_counts = {}
    total_word_count = 0

    for t in cluster.text_data:
        words = filter(lambda x: len(x) > 4, t[0][0].split(' ')[2:])
        for w in words:
            if w == '':
                continue
                
            
            if w in word_counts:
                word_counts[w] += 1
            else:
                word_counts[w] = 1

            total_word_count += 1

    entropy = 0
    for t in cluster.text_data:
        idf_sum = 0
        msg_entropy = 0
        words = filter(lambda x: len(x) > 4, t[0][0].split(' ')[2:])
        for w in words:
            idf = idfs.get(w, 1)
            #for w, c in word_counts.iteritems():
            c = word_counts[w]
            prob = float(c) / total_word_count
            msg_entropy += -idf * math.log(prob)/math.log(2)/float(len(cluster.text_data))
            idf_sum += idf
            
        msg_entropy /= idf_sum
        entropy += msg_entropy

    return entropy


