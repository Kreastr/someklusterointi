#!/usr/bin/env python

from __future__ import print_function
import sys
from getopt import GetoptError, getopt
import pickle
from operator import itemgetter
import time
from datetime import datetime
from datetime import timedelta
import random
import math
import simplejson

import numpy as np
from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections
from nearpy.distances import CosineDistance

sys.path.append('./swivel')
from vecs import Vecs
from calculate_idfs import calculate_idfs
import cluster_exporter
from sentiment import getSentiment
import construct_translation_mat

# Maximum distance for clustering
CLUSTER_THRESHOLD = 0.8
# Minimum entropy before a cluster is classified as spam
ENTROPY_THRESHOLD = 3.5

# TODO implement better system
# offset finnish cluster id's to avoid id conflicts
FI_CLUSTER_ID_OFFSET = 10000000

# Locality senstive hashing parameters, chosen based on the paper 'Streaming First Story Detection with applicaiton to Twitter'
HYPERPLANE_COUNT  = 13
HASH_LAYERS       = 7
lsh_distance_func = CosineDistance() # 1 - cos(a)

try:
    opts, args = getopt(sys.argv[1:], 'v:e:t:i:l:', ['vocab=', 'embeddings=', 'text=', 'idfs=', 'lang='])
except GetoptError as e:
    print(e, file=sys.stderr)
    #sys.exit(2)

opt_vocab = 'vocab.txt'
opt_embeddings = 'vecs.bin'
opt_text = 'tweet_replies_non_alpha_true-ru_lem.txt'
opt_idfs = 'tweet_idfs.json'
opt_lang = 'ru'

for o, a in opts:
  if o in ('-v', '--vocab'):
    opt_vocab = a
  if o in ('-e', '--embeddings'):
    opt_embeddings = a
  if o in ('-t', '--text'):
    opt_text  = a
  if o in ('-i', '--idfs'):
    opt_idfs = a
  if o in ('-l', '--lang'):
    opt_lang = a

class Cluster:
    def __init__(self, id, lang):
        self.id        = id
        self.center    = None
        self.norm      = None
        self.documents = []
        self.text      = []

        self.lang = lang

        self.last_update = 0
        self.created_at  = 0

        self.hourly_growth_rate = []

        self.hourly_keywords = []

        self.hourly_sentiment   = []
        self.hourly_accum_sentiment   = []

        self.last_size = 0
        # when growth was calculated for the first time
        self.first_growth_time = -1


# Every new cluster gets an unique id which is the key for this dictionary
clusters = {}
next_cluster_id = FI_CLUSTER_ID_OFFSET if opt_lang == 'fi' else 0

vecs = Vecs(opt_vocab, opt_embeddings)
# Locality Sensitive Hashing
lsh_engine = Engine(vecs.dim, lshashes=[RandomBinaryProjections('rpb', HYPERPLANE_COUNT) for i in range(HASH_LAYERS)], distance=lsh_distance_func)

# Return closest clusters to a given sentence
def query_clusters(query, idfs):

    doc_vec = document_to_vector(query.split(' '), idfs)

    if doc_vec is None:
        return None

    return sorted([(1 - doc_vec.dot(c.center) / c.norm, c) for id, c in clusters.iteritems()])


# Returns sorted list of (distance, document text) tuples
def doc_distances_to_center(cluster):
    distances = []

    for i in range(len(cluster.text)):
        dist = 1 - cluster.documents[i].dot(cluster.center) / cluster.norm
        distances.append((dist, cluster.text[i]))

    return sorted(distances)


def document_to_vector(word_list, idfs):
    use_idf_weighting = idfs is not None
    word_vecs = []
    idf_sum = 0

    for word in word_list:
        word = word.strip()
        if word == '':
            continue

        vec = vecs.lookup(word)

        if vec is None:
            continue

        if use_idf_weighting:
            idf = idfs.get(word, 1)
            idf_sum += idf

            word_vecs.append((vec, idf))
        else:
            word_vecs.append(vec)


    if len(word_vecs) == 0:
        return None

    # calculate document vector
    if use_idf_weighting:
        doc_vec = word_vecs[0][0] * word_vecs[0][1] / idf_sum
        for word_vec in word_vecs[1:]:
            doc_vec += word_vec[0] * word_vec[1] / idf_sum

    else:
        if len(word_vecs) > 1:
            doc_vec = np.mean(word_vecs, axis=0)
        else:
            doc_vec = word_vecs[0]

    return doc_vec


# Every line in the input file should start with a timestamp in ms and id of document,
# followed by the whole document, all separated with spaces and without newlines.
#
# Note: minimum word frequency is often implemented by the vector model already
def construct_clusters(filename, from_line=0, from_date=None, idfs=None, lang=None):
    global next_cluster_id

    if lang != 'ru' and lang != 'fi':
        print("Lang must be 'ru' or 'fi'")
        return

    tweet_file = open(filename)

    try:
        line = 0

        # performance counter
        last_print_line = 0
        last_print_time = time.time()

        # used for calculating hourly growth in tweet time
        last_growth_calc = 0

        for tweet in tweet_file:

            line += 1
            if line < from_line:
                continue

            tweet_parts = tweet.strip().split(' ')
            try:
                tweet_time  = int(tweet_parts[0])
            except ValueError:
                print('Invalid document on line %d: %s' % (line, tweet))
                continue

            if from_date is not None and datetime.utcfromtimestamp(tweet_time * 0.001) < from_date:
                continue

            # TEMP ignore gameinsight spam and short tweets
            if len(tweet_parts) < 6 or tweet.find('gameinsight') != -1:
                continue


            # remove really old clusters with a small amount of documents
            if line % 100000 == 0:
                to_be_removed = []
                for k, c in clusters.iteritems():
                    if line - c.last_update > (100000 * len(c.documents)) and len(c.documents) < 10:
                        to_be_removed.append((k, c.center))

                for t in to_be_removed:
                    lsh_engine.delete_vector(t[0], t[1])
                    clusters.pop(t[0])

                if len(to_be_removed) > 0:
                    print("Removed %d stagnant clusters" % len(to_be_removed))

            # save periodically
            if False:#line % 1000000 == 0 and line != 0:
                save_results(filename + '_' + str(line))

            # print status
            if line % 1000 == 0:
                new_time = time.time()
                print("Line: %d, Date: %s, Clusters: %d, %d lines/s" % (line, datetime.utcfromtimestamp(tweet_time / 1000), len(clusters), int((line - last_print_line) / (new_time - last_print_time))))
                last_print_line = line
                last_print_time = new_time


            # calculate growth rate
            time_since_last_growth = tweet_time - last_growth_calc
            if time_since_last_growth > 1000 * 60 * 60:
                last_growth_calc = tweet_time

                for id, c in clusters.iteritems():

                    if (c.created_at < 1405555200000): # 17/07/2014 00:00:00
                        continue

                    # calculate growth for first 12h
                    if len(c.hourly_growth_rate) < 12:
                        growth_rate = (len(c.text) - c.last_size) / float(time_since_last_growth) * 1000 * 60 * 60
                        if len(c.hourly_growth_rate) == 0:
                            c.first_growth_time = tweet_time

                        c.hourly_growth_rate.append(growth_rate)

                        # calculate sentiment for new tweets
                        if len(c.documents) > c.last_size:
                            cluster_vector = np.mean(c.documents[c.last_size:], axis=0)
                            sentiment = getSentiment(cluster_vector)
                        else:
                            sentiment = 0

                        c.hourly_sentiment.append(sentiment)

                        # calculate total sentiment so far
                        sentiment = getSentiment(np.mean(c.documents, axis=0))
                        c.hourly_accum_sentiment.append(sentiment)

                        c.last_size = len(c.text)
                        c.hourly_keywords.append(cluster_exporter.get_keywords(c, idfs)[:3])


                        # print quickly growing ones with high enough entropy
                        if growth_rate < 10:
                            continue

                        entropy = cluster_exporter.calculate_cluster_entropy(c)
                        if entropy < ENTROPY_THRESHOLD:
                            continue

                        print('Quickly growing cluster %d: %d tweets, %d tweets/h, entropy %.2f\n' % (id, len(c.text), int(growth_rate), entropy))
                        print('\n'.join(random.sample(c.text, min(len(c.text), 8))))
                        print('\n\n')


            doc_vec = document_to_vector(tweet_parts[2:], idfs)

            if doc_vec is None:
                continue

            # look for nearest cluster
            lowest_index = -1
            nearest_neighbours = lsh_engine.neighbours(doc_vec)
            if len(nearest_neighbours) > 0:
                # get closest one from tuple (cluster vector, cluster index, distance)
                nn = min(nearest_neighbours, key=itemgetter(2))

                if nn[2] < CLUSTER_THRESHOLD:
                    lowest_index = nn[1]

            if lowest_index != -1:
                c = clusters[lowest_index]
                c.documents.append(doc_vec)
                c.text.append(tweet.strip())
                c.last_update = line


                # update the cluster center if the cluster is small
                if len(c.documents) < 100:
                    lsh_engine.delete_vector(lowest_index, c.center)

                    c.center = np.mean(c.documents, axis=0)
                    c.norm   = np.linalg.norm(c.center)

                    lsh_engine.store_vector(c.center, lowest_index)
            else:
                # no cluster found, construct new one
                c = Cluster(next_cluster_id, lang)
                c.center = doc_vec
                c.norm = np.linalg.norm(c.center)

                c.documents = [doc_vec]
                c.text = [tweet.strip()]

                c.last_update = line
                c.created_at = tweet_time
                c.last_growth_calc = tweet_time

                lsh_engine.store_vector(doc_vec, next_cluster_id)
                clusters[next_cluster_id] = c
                next_cluster_id += 1

    except KeyboardInterrupt:
        print("Line: %d Clusters: %d" % (line, len(clusters)))
        print("Cancelled")

def merge_clusters(a, b):
    a.text.append(b.text)
    a.documents.extend(b.documents)
    for i in range(min(len(a.hourly_growth_rate), len(b.hourly_growth_rate))):
        a.hourly_growth_rate[i] += b.hourly_growth_rate[i]

def merge_close_clusters():
    cluster_ids = clusters.keys()
    clusters_to_remove = []

    for i in range(len(clusters) - 1):
        a = clusters[cluster_ids[i]]
        for j in range(i + 1, len(clusters)):
            if cluster_ids[j] not in clusters_to_remove:
                b = clusters[cluster_ids[j]]
                dist = 1 - a.center.dot(b.center) / (a.norm * b.norm)
                print(dist)
                if dist < CLUSTER_THRESHOLD:
                    merge_clusters(a, b)
                    clusters_to_remove.append(cluster_ids[j])

    print('Merged %d clusters' % len(clusters_to_remove))
    for id in clusters_to_remove:
        lsh_engine.delete_vector(id, clusters[id].center)
        del clusters[id]

def save_results(filesuffix):
    with open('clusters_%s.bin' % filesuffix,'wb') as f:
        pickle.dump(clusters, f)

def load_results(filesuffix):
    global clusters

    with open('clusters_%s.bin' % filesuffix,'rb') as f:
        clusters = pickle.load(f)

def main():
    # TODO save result instead of recalculating
    if opt_lang == 'fi':
        translation = construct_translation_mat.WordVectorTranslator()
        for i in range(len(vecs.vecs)):
            vecs.vecs[i] = translation.translate_vec(vecs.vecs[i])

    global idfs
    print('Loading idf')
    idfs = calculate_idfs(opt_idfs, force_recalc=False)
    print('Calculating clusters')
    construct_clusters(opt_text, from_date=datetime(2014, 7, 14), idfs=idfs, lang=opt_lang)

if __name__ == "__main__":
    main()
