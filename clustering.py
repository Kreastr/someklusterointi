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
from nearpy.distances import EuclideanDistance, CosineDistance

sys.path.append('./swivel')
from vecs import Vecs
from calculate_idfs import calculate_idfs
import cluster_exporter
from sentiment import getSentiment
import construct_translation_mat
from dateutil import parser

from flask import Flask,send_from_directory,request
flask_app = Flask("Tweet Analysis")

from scipy.stats  import normaltest
from sklearn.cluster import KMeans    
from pytz import timezone
import pytz

utc = pytz.utc

# Maximum distance for clustering
CLUSTER_THRESHOLD = 0.8*30
# Minimum entropy before a cluster is classified as spam
ENTROPY_THRESHOLD = 3.5

# TODO implement better system
# offset Finnish cluster ids to avoid id conflicts
FI_CLUSTER_ID_OFFSET = 10000000

# Locality senstive hashing parameters, chosen based on the paper 'Streaming First Story Detection with applicaiton to Twitter'
HYPERPLANE_COUNT  = 13
HASH_LAYERS       = 14
lsh_distance_func = EuclideanDistance()#CosineDistance() # 1 - cos(a)

try:
    opts, args = getopt(sys.argv[1:], 'v:e:t:i:l:', ['vocab=', 'embeddings=', 'text=', 'idfs=', 'lang='])
except GetoptError as e:
    print(e, file=sys.stderr)
    #sys.exit(2)

opt_vocab = 'vocab.txt'
opt_embeddings = 'vecs.bin'
opt_text = 'starting_from_2014_07_16.txt'#'tweet_replies_non_alpha_true-ru_lem.txt'
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
    def __init__(self, id, lang, power=1.0):
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

        # power defines size of the cluster threshold and its ability to attract new messages
        self.power = power
        self.last_size = 0
        # when growth was calculated for the first time
        self.first_growth_time = -1

vecs = Vecs(opt_vocab, opt_embeddings)

print('Loading idf')
idfs = calculate_idfs(opt_idfs, force_recalc=False)

# Returns a sorted list of the cluster's documents as (distance, document text) tuples
# where distance is the distance between the document's vector and the cluster center.
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

class ClusterAnalyser:
    def __init__(self):
       self.resetClusters()
        
    def resetClusters(self):
        # Every new cluster gets an unique id which is the key for this dictionary
        self.clusters = {}
        self.next_cluster_id = FI_CLUSTER_ID_OFFSET if opt_lang == 'fi' else 0
        # Locality Sensitive Hashing
        self.lsh_engine = Engine(vecs.dim, lshashes=[RandomBinaryProjections('rpb', HYPERPLANE_COUNT) for i in range(HASH_LAYERS)], distance=lsh_distance_func)

    # Returns closest clusters to a given sentence, in a sorted list of (distance, cluster) tuples.
    def query_clusters(query, idfs):

        doc_vec = document_to_vector(query.split(' '), idfs)

        if doc_vec is None:
            return None

        return sorted([(1 - doc_vec.dot(c.center) / c.norm, c) for id, c in self.clusters.iteritems()])


    # look for nearest cluster
    def lookupNearest(self, doc_vec):
        lowest_index = -1
        nearest_neighbours = self.lsh_engine.neighbours(doc_vec)
        if len(nearest_neighbours) > 0:
            # get closest one from tuple (cluster vector, cluster index, distance)
            nn = min(nearest_neighbours, key=lambda x: (x[2]/self.clusters[x[1]].power))

            if nn[2] < (CLUSTER_THRESHOLD*self.clusters[nn[1]].power):
                lowest_index = nn[1]
        return lowest_index
        

    def initNewCluster(self, doc_vec, tweet, line, tweet_time, lang):
        
        c = Cluster(self.next_cluster_id, lang, power=1.0)
        c.center = np.mean(doc_vec, axis=0)
        c.norm = np.linalg.norm(c.center)

        c.documents = doc_vec
        c.text = tweet

        c.last_update = line
        c.created_at = tweet_time
        c.last_growth_calc = tweet_time

        self.lsh_engine.store_vector(c.center, self.next_cluster_id)
        self.clusters[self.next_cluster_id] = c
        self.next_cluster_id += 1

    # Every line in the input file should start with a timestamp in ms and id of document,
    # followed by the whole document, all separated with spaces and without newlines.
    #
    # Note: minimum word frequency is often implemented by the vector model already
    def construct_clusters(self, filename, from_line=0, from_date=None, to_date=None,idfs=None, lang=None):

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
                
                tweet_time_notz = datetime.utcfromtimestamp(tweet_time * 0.001)
                tweet_time_utc = utc.localize(tweet_time_notz)
                
                if from_date is not None and tweet_time_utc < from_date:
                    continue
                    
                if to_date is not None and tweet_time_utc > to_date:
                    break
                    
                # TEMP ignore gameinsight spam and short tweets
                if len(tweet_parts) < 6 or tweet.find('gameinsight') != -1:
                    continue


                # remove really old clusters with a small amount of documents
                if line % 100000 == 0:
                    to_be_removed = []
                    for k, c in self.clusters.iteritems():
                        if line - c.last_update > (100000 * len(c.documents)) and len(c.documents) < 10:
                            to_be_removed.append((k, c.center))

                    for t in to_be_removed:
                        self.lsh_engine.delete_vector(t[0], t[1])
                        self.clusters.pop(t[0])

                    if len(to_be_removed) > 0:
                        print("Removed %d stagnant clusters" % len(to_be_removed))

                # save periodically
                if False:#line % 1000000 == 0 and line != 0:
                    save_results(filename + '_' + str(line))

                # print status
                if line % 1000 == 0:
                    new_time = time.time()
                    print("Line: %d, Date: %s, Clusters: %d, %d lines/s" % (line, tweet_time_notz, len(self.clusters), int((line - last_print_line) / (new_time - last_print_time))))
                    last_print_line = line
                    last_print_time = new_time


                # calculate growth rate
                time_since_last_growth = tweet_time - last_growth_calc
                if time_since_last_growth > 1000 * 60 * 60:
                    last_growth_calc = tweet_time

                    for id, c in self.clusters.iteritems():

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
                            c.hourly_keywords.append(cluster_exporter.get_keywords(c, idfs)[:3])#['three','random','words']


                            # print quickly growing ones with high enough entropy
                            #if growth_rate < 10:
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


                    
                lowest_index = self.lookupNearest(doc_vec)
                
                if lowest_index != -1:
                    c = self.clusters[lowest_index]
                    c.documents.append(doc_vec)
                    c.text.append(tweet.strip())
                    c.last_update = line


                    # update the cluster center if the cluster is small
                    if len(c.documents) < 100:
                        self.lsh_engine.delete_vector(lowest_index, c.center)

                        c.center = np.mean(c.documents, axis=0)
                        c.norm   = np.linalg.norm(c.center)

                        self.lsh_engine.store_vector(c.center, lowest_index)
                else:
                    # no cluster found, construct new one
                    self.initNewCluster([doc_vec], [tweet.strip()], line, tweet_time, lang)
                                        
                if line % 10000 == 0:
                    for c_idx, c in list(self.clusters.iteritems()): 
                        #print ([c_idx, c])        
                        if len(c.documents) > 60:  
                            def proposeKMeansSplit(X, txt):
                                kmeans = KMeans(n_clusters=2, random_state=(line+c_idx)).fit(X)
                                lbls = kmeans.labels_
                                z = zip(X, lbls)
                                z2 = zip(txt, lbls)
                                return zip(list(map(lambda x: x[0], filter(lambda x: x[1] == 0, z))), list(map(lambda x: x[0], filter(lambda x: x[1] == 0, z2)))),zip( list(map(lambda x: x[0], filter(lambda x: x[1] == 1, z))), list(map(lambda x: x[0], filter(lambda x: x[1] == 1, z2))))
                            
                            def computeNormalLikelyhood(pool):
                                return pow(reduce(lambda x,y: x*y, normaltest(list(map(lambda x: x[0], pool)), axis=0)[1].tolist(),1.0), 1.0/300)
                            
                            if random.random() < 0.5:
                                c.center = np.mean(c.documents, axis=0)
                                a, b = proposeKMeansSplit(c.documents, c.text)
                                if len(a) < 20:
                                    continue
                                if len(b) < 20:
                                    continue
                                
                                probJoin = computeNormalLikelyhood(zip(c.documents, c.text))
                                probSplit = computeNormalLikelyhood(a)*computeNormalLikelyhood(b)
                                if probJoin < probSplit:
                                    c.documents = list(map(lambda x: x[0],a))
                                    c.texts = list(map(lambda x: x[0],a))
                                    self.lsh_engine.delete_vector(lowest_index, c.center)
                                    c.center = np.mean(c.documents, axis=0)
                                    c.norm   = np.linalg.norm(c.center)
                                    self.lsh_engine.store_vector(c.center, lowest_index)
                                    # copy time parameters for now
                                    print ("Split cluster %d into %d and %d" % (c_idx, len(a), len(b)))
                                    self.initNewCluster(list(map(lambda x: x[0],b)), list(map(lambda x: x[1],b)), c.last_update, c.created_at, lang)
                            else:
                                # Test message redistribution
                                nearest_neighbour_clusters = self.lsh_engine.neighbours(c.center)
                                if len(nearest_neighbour_clusters) > 1:
                                    
                                    # save old value
                                    power_before = c.power
                                    
                                    # gather all messages from affected clusters 
                                    message_pool = []
                                    new_pools_incr = dict()
                                    new_pools_decr = dict()
                                    for nn in nearest_neighbour_clusters:
                                        cluster_nn = self.clusters[nn[1]]
                                        if len(cluster_nn.documents) > 40:
                                            new_pools_incr[nn[1]] = list()
                                            new_pools_decr[nn[1]] = list()
                                            for i in range(len(cluster_nn.documents)):
                                                message_pool.append((cluster_nn.documents[i],cluster_nn.text[i], i))
                                    
                                    # put messages in incremented set with target cluster's power incremented     
                                    c.power = power_before * 1.1
                                    
                                    for m in message_pool:
                                        lowest_index = self.lookupNearest(m[0])
                                        if lowest_index in new_pools_incr:
                                            new_pools_incr[lowest_index].append(m)
                                    
                                    # put messages in incremented set with target cluster's power decremented
                                    c.power = power_before / 1.1
                                    for m in message_pool:
                                        lowest_index = self.lookupNearest(m[0])
                                        if lowest_index in new_pools_decr:
                                            new_pools_decr[lowest_index].append(m)   
                                    
                                    # compute normal distribution probabilities
                                    prob_incr = 1.0
                                    prob_decr = 1.0
                                    
                                    for poolidx, pool in new_pools_incr.iteritems():
                                        if len(pool) > 7:
                                            prob_incr *= computeNormalLikelyhood(pool)
                                        else:
                                            prob_incr *= 0.01
                                    
                                    for poolidx, pool in new_pools_decr.iteritems():
                                        if len(pool) > 7:
                                            prob_decr *= computeNormalLikelyhood(pool)
                                        else:
                                            prob_decr *= 0.01
                                    
                                    # update power and messages                 
                                    c.power = (power_before * 1.1) if prob_incr > prob_decr else (power_before / 1.1)
                                    new_clusters = new_pools_incr if prob_incr > prob_decr else new_pools_decr
                                    for poolidx, pool  in new_clusters.iteritems():
                                        self.clusters[poolidx].documents = list(map(lambda x: x[0], pool))
                                        self.clusters[poolidx].text = list(map(lambda x: x[1], pool))
                                


        except KeyboardInterrupt:
            print("Line: %d Clusters: %d" % (line, len(self.clusters)))
            print("Cancelled")
 
 
def get_clusters():
    day = request.args.get('day')
    dt = parser.parse(day)
    analyser = ClusterAnalyser()
    analyser.construct_clusters(opt_text, from_date=(dt-timedelta(hours=12)), to_date=(dt+timedelta(days=1)), idfs=idfs, lang=opt_lang)
    
    return simplejson.dumps(cluster_exporter.convert_to_dict(analyser.clusters,idfs, None))
        
def send_index(path):
    return send_from_directory('visualisation', path)
    
def main():
    # TODO save result instead of recalculating
    if opt_lang == 'fi':
        translation = construct_translation_mat.WordVectorTranslator()
        for i in range(len(vecs.vecs)):
            vecs.vecs[i] = translation.translate_vec(vecs.vecs[i])


    print('Calculating clusters')
    #construct_clusters(opt_text, from_date=datetime(2014, 7, 14), idfs=idfs, lang=opt_lang)#, to_date=datetime(2014, 7, 20)
    
    flask_app.add_url_rule('/cluster_data_test.json', 'get_clusters', get_clusters) 
    flask_app.add_url_rule('/<path:path>', 'send_index', send_index)  
    flask_app.run(host='0.0.0.0',port='80', 
            debug = True, use_reloader=False)#, ssl_context=context

if __name__ == "__main__":
    main()
