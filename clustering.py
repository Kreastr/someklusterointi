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
import cProfile, pstats, StringIO

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
pr = cProfile.Profile()
utc = pytz.utc

analysers = dict();
from sklearn.manifold import TSNE
# Maximum distance for clustering
CLUSTER_THRESHOLD = 1
# Minimum entropy before a cluster is classified as spam
ENTROPY_THRESHOLD = 3.5

# TODO implement better system
# offset Finnish cluster ids to avoid id conflicts
FI_CLUSTER_ID_OFFSET = 10000000

# Locality senstive hashing parameters, chosen based on the paper 'Streaming First Story Detection with applicaiton to Twitter'
HYPERPLANE_COUNT  = 10
HASH_LAYERS       = 8
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
        self.text_data      = []

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
        
    def updatePower(self):
        if len(self.documents) > 5:
            self.power = np.mean(np.std(self.documents, axis=0))
            
            
    def calculateGrowthAndSentiment(self):
        
        mintime = min(map(lambda x: x[1], self.text_data))
        self.created_at = mintime
        self.first_growth_time = mintime
        self.hourly_growth_rate = [0]
        self.hourly_sentiment = [0]
        self.hourly_accum_sentiment = [0]
        self.hourly_keywords = [['']]
        self.hourly_tags = [['']]
        hourly_tweets = dict()
        hourly_tweets_accum = dict()
        maxidx=0
        for tw in zip(self.documents, self.text_data):
            idx = int((tw[1][1]-mintime)/3600/1000)
            if idx > maxidx:
                maxidx = idx
                
            while idx >= len(self.hourly_growth_rate):
                self.hourly_growth_rate.append(0)
                self.hourly_sentiment.append(0)
                self.hourly_accum_sentiment.append(0)
                self.hourly_keywords.append([''])
                self.hourly_tags.append([''])
                
            self.hourly_growth_rate[idx] += 1
            if not idx in hourly_tweets:
                hourly_tweets[idx] = []
            hourly_tweets[idx].append(tw)
            for i in range(idx):
                if not idx in hourly_tweets_accum:
                    hourly_tweets_accum[idx] = []
                hourly_tweets_accum[idx].append(tw)
            
        for idx in range(maxidx+1):
            if idx in hourly_tweets_accum:
                hta = list(map(lambda x: x[1],(hourly_tweets_accum[idx])))
                self.hourly_keywords[idx]=(cluster_exporter.get_keywords_for_message_list(hta, idfs)[:3])
                self.hourly_tags[idx]=cluster_exporter.getTagsForTexts(self, hta)[:3]
             
            if idx in hourly_tweets:    
                cluster_vector = np.mean(list(map(lambda x: x[0],(hourly_tweets[idx]))), axis=0)
                self.hourly_sentiment[idx]=getSentiment(cluster_vector)  
                self.hourly_growth_rate[idx] = len(hourly_tweets[idx])
                
            if idx in hourly_tweets_accum:
                cluster_vector = np.mean(list(map(lambda x: x[0],(hourly_tweets_accum[idx]))), axis=0)
                self.hourly_accum_sentiment[idx]=getSentiment(cluster_vector)  
            

vecs = Vecs(opt_vocab, opt_embeddings)

print('Loading idf')
idfs = calculate_idfs(opt_idfs, force_recalc=False)

# Returns a sorted list of the cluster's documents as (distance, document text) tuples
# where distance is the distance between the document's vector and the cluster center.
def doc_distances_to_center(cluster):
    distances = []

    for i in range(len(cluster.text_data)):
        dist = 1 - cluster.documents[i].dot(cluster.center) / cluster.norm
        distances.append((dist, cluster.text_data[i]))

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

def proposeKMeansSplit(X, txt, line, c_idx):
    kmeans = KMeans(n_clusters=2, random_state=(line+c_idx)).fit(X)
    lbls = kmeans.labels_
    z = zip(X, lbls)
    z2 = zip(txt, lbls)
    return zip(list(map(lambda x: x[0], filter(lambda x: x[1] == 0, z))), list(map(lambda x: x[0], filter(lambda x: x[1] == 0, z2)))),zip( list(map(lambda x: x[0], filter(lambda x: x[1] == 1, z))), list(map(lambda x: x[0], filter(lambda x: x[1] == 1, z2))))

def computeNormalLikelyhood(pool):
    return pow(reduce(lambda x,y: x*y, normaltest(list(map(lambda x: x[0], pool)), axis=0)[1].tolist(),1.0), 1.0/300)


def getSplitsWorker(c_idx, c, min_update):
    if len(c.documents) < 120:  
        return (c_idx, dict(result=False))
        
    if c.last_update < min_update:#line - self.TUNE_INTERVAL
        return (c_idx, dict(result=False))
            
    c.center = np.mean(c.documents, axis=0)
    a, b = proposeKMeansSplit(c.documents, c.text_data, min_update, c_idx)
    if len(a) < 20:
        return (c_idx, dict(result=False))
    if len(b) < 20:
        return (c_idx, dict(result=False))
    
    probJoin = computeNormalLikelyhood(zip(c.documents, c.text_data))
    probSplit = computeNormalLikelyhood(a)*computeNormalLikelyhood(b)
    if probJoin < probSplit:
        return (c_idx, dict(result=True, a=a, b=b,probSplit=probSplit,probJoin=probJoin))
    else:
        return (c_idx, dict(result=False))
                
def doAnalyseSplit(x):
    return getSplitsWorker(x[0][0],x[0][1],x[1])
    
class ClusterAnalyser:
    def __init__(self):
       from multiprocessing import Pool
       self.resetClusters()
       self.TUNE_INTERVAL = 5000
       self.ncnttot=0
       self.ncntq=0
       self.store_cnt = 20
       self.store_join_cnt = 20
       self.p = Pool(20)
        
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
        ncnt = self.lsh_engine.candidate_count(doc_vec)
        self.ncnttot += ncnt
        self.ncntq += 1
        
        nearest_neighbours = self.lsh_engine.neighbours(doc_vec)
        if len(nearest_neighbours) > 0:
            # get closest one from tuple (cluster vector, cluster index, distance)
            nn = min(nearest_neighbours, key=lambda x: (x[2]/self.clusters[x[1]].power))

            #if nn[2] < (CLUSTER_THRESHOLD*self.clusters[nn[1]].power):
            lowest_index = nn[1]
        return lowest_index
        

    def initNewCluster(self, doc_vec, tweet, line, tweet_time, lang, tweet_post_time):
        
        c = Cluster(self.next_cluster_id, lang, power=1.0)
        c.center = np.mean(doc_vec, axis=0)
        c.norm = np.linalg.norm(c.center)

        c.documents = doc_vec
        c.text_data = zip(tweet,tweet_post_time)

        c.last_update = line
        c.created_at = tweet_time
        c.last_growth_calc = tweet_time

        self.lsh_engine.store_vector(c.center, self.next_cluster_id)
        self.clusters[self.next_cluster_id] = c
        self.next_cluster_id += 1
        c.updatePower()
        
 


    def tuneClusters(self):
        line = self.line
        deleted_clusters = []
        print ('parallel preprocessing ... ')
        #parallel preprocessing 
        dlist = list(self.clusters.iteritems())
        params = [self.line - self.TUNE_INTERVAL]*len(dlist)
        split_test_out = dict(self.p.map(doAnalyseSplit, zip(dlist, params)))
        

        print ('done')
        for c_idx, c in list(self.clusters.iteritems()): 
            if c_idx in deleted_clusters:
                continue
            #print ([c_idx, c])        
            if c.last_update < line - self.TUNE_INTERVAL:
                continue
                
            if len(c.documents) > 60:  
                
                if split_test_out[c_idx]['result']:
                    a = split_test_out[c_idx]['a']
                    b = split_test_out[c_idx]['b']
                    probJoin = split_test_out[c_idx]['probJoin']
                    probSplit = split_test_out[c_idx]['probSplit']
                    c.documents = list(map(lambda x: x[0],a))
                    c.text_data = list(map(lambda x: x[1],a))
                    self.lsh_engine.delete_vector(c_idx)
                    c.center = np.mean(c.documents, axis=0)
                    c.norm   = np.linalg.norm(c.center)
                    c.updatePower()
                    self.lsh_engine.store_vector(c.center, c_idx)
                    # copy time parameters for now
                    print ("Split cluster %d into %d and %d" % (c_idx, len(a), len(b)))
                    self.initNewCluster(list(map(lambda x: x[0],b)), list(map(lambda x: x[1][0],b)), c.last_update, c.created_at, c.lang, list(map(lambda x: x[1][1],b)))
                    if self.store_cnt > 0:
                        pickle.dump(dict(a=a,b=b,probJoin=probJoin,probSplit=probSplit),open('stored_split_cases_%d.pckl'%self.store_cnt,'wb'))
                        self.store_cnt -= 1
                
                # Test merge with random nearest
                nearest_neighbour_clusters = self.lsh_engine.neighbours(c.center)
                if len(nearest_neighbour_clusters) > 1:
                    ann, bnn = random.sample(nearest_neighbour_clusters, 2)
                    
                    a= zip(self.clusters[ann[1]].documents, self.clusters[ann[1]].text_data)
                    b= zip(self.clusters[bnn[1]].documents, self.clusters[bnn[1]].text_data)
                    if len(a) < 20:
                        continue
                    if len(b) < 20:
                        continue
                    if self.clusters[ann[1]].lang != self.clusters[bnn[1]].lang:
                        continue
                        
                    probJoin = computeNormalLikelyhood(a+b)
                    probSplit = computeNormalLikelyhood(a)*computeNormalLikelyhood(b)
                    if probJoin > probSplit:
                         deleted_clusters.append(ann[1])
                         deleted_clusters.append(bnn[1])
                         print ("Join clusters %d (%d) and %d (%d) %f > %f" % (ann[1], len(a), bnn[1], len(b), probJoin, probSplit))
                         if self.store_join_cnt > 0:
                             pickle.dump(dict(a=a,b=b,probJoin=probJoin,probSplit=probSplit),open('stored_join_cases_%d.pckl'%self.store_join_cnt,'wb'))
                             self.store_join_cnt -= 1
                         self.initNewCluster(list(map(lambda x: x[0],a+b)), list(map(lambda x: x[1][0],a+b)), max(self.clusters[bnn[1]].last_update,self.clusters[ann[1]].last_update), max(self.clusters[bnn[1]].created_at,self.clusters[ann[1]].created_at), self.clusters[ann[1]].lang, list(map(lambda x: x[1][1],a+b)))
                         self.lsh_engine.delete_vector(ann[1])
                         self.clusters.pop(ann[1])
                         self.lsh_engine.delete_vector(bnn[1])
                         self.clusters.pop(bnn[1])
                                 
                                 
                  
                                
    def purgeClusters(self):
        line = self.line
        to_be_removed = []
        for k, c in self.clusters.iteritems():
            if line - c.last_update > (100000 * len(c.documents)) and len(c.documents) < 10:
                to_be_removed.append((k, c.center))

        for t in to_be_removed:
            self.lsh_engine.delete_vector(t[0])
            self.clusters.pop(t[0])

        if len(to_be_removed) > 0:
            print("Removed %d stagnant clusters" % len(to_be_removed))

    def calcGrowthRate(self):
        line = self.line
        tweet_time = self.tweet_time
        time_since_last_growth = self.time_since_last_growth
        for id, c in self.clusters.iteritems():
            #if (c.created_at < 1405555200000): # 17/07/2014 00:00:00
            #    continue

            c.calculateGrowthAndSentiment()
            
            ## calculate growth for first 12h
            #if len(c.hourly_growth_rate) < 12:
                #growth_rate = (len(c.text_data) - c.last_size) / float(time_since_last_growth) * 1000 * 60 * 60
                #if len(c.hourly_growth_rate) == 0:
                    #c.first_growth_time = tweet_time

                #c.hourly_growth_rate.append(growth_rate)

                ## calculate sentiment for new tweets
                #if len(c.documents) > c.last_size:
                    #cluster_vector = np.mean(c.documents[c.last_size:], axis=0)
                    #sentiment = getSentiment(cluster_vector)
                #else:
                    #sentiment = 0

                #c.hourly_sentiment.append(sentiment)

                ## calculate total sentiment so far
                #sentiment = getSentiment(np.mean(c.documents, axis=0))
                #c.hourly_accum_sentiment.append(sentiment)

                #c.last_size = len(c.text_data)
                #c.hourly_keywords.append(cluster_exporter.get_keywords(c, idfs)[:3])#['three','random','words']


                ## print quickly growing ones with high enough entropy
                ##if growth_rate < 10:
                #continue
                
                #entropy = cluster_exporter.calculate_cluster_entropy(c)
                #if entropy < ENTROPY_THRESHOLD:
                    #continue

                #print('Quickly growing cluster %d: %d tweets, %d tweets/h, entropy %.2f\n' % (id, len(c.text_data), int(growth_rate), entropy))
                #print('\n'.join(list(map(lambda x: x[0],random.sample(c.text_data, min(len(c.text), 8))))))
                #print('\n\n')
        
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
            self.line = 0
            

            # performance counter
            self.last_print_line = 0
            self.last_print_time = time.time()

            # used for calculating hourly growth in tweet time
            self.last_growth_calc = 0
            self.tweet_time = 0
            
            self.tweet_time_notz = datetime.utcfromtimestamp(0)
                
            for tweet in tweet_file:

                self.line += 1
                
                if self.line < from_line:
                    continue
                                    
                if self.line % self.TUNE_INTERVAL == 0:
                    #pr.disable()
                    self.tuneClusters()
                    #pr.enable()
                    

                    
                # save periodically
                if False:#self.line % 1000000 == 0 and self.line != 0:
                    save_results(filename + '_' + str(self.line))    
                                   
                # remove really old clusters with a small amount of documents
                if self.line % 100000 == 0:
                    self.purgeClusters()
                    
                # print status
                if self.line % 1000 == 0:
                    #pr.disable()
                    new_time = time.time()
                    print("Line: %d, Date: %s, Clusters: %d, %d lines/s AVG candidates: %d" % (self.line, self.tweet_time_notz, len(self.clusters), int((self.line - self.last_print_line) / (new_time - self.last_print_time)), int(self.ncnttot/(self.ncntq+0.0000001))))
                    #if  int((self.line - self.last_print_line) / (new_time - self.last_print_time)) < 50:
                    #    s = StringIO.StringIO()
                    #    sortby = 'cumulative'
                    #    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
                    #    ps.print_stats()
                    #    print (s.getvalue())
                    self.last_print_line = self.line
                    self.last_print_time = new_time
                    self.ncnttot = 0
                    self.ncntq = 0
                    #pr.enable()


                # calculate growth rate
                #self.time_since_last_growth = self.tweet_time - self.last_growth_calc
                #if self.time_since_last_growth > 1000 * 60 * 60:
                #    self.last_growth_calc = self.tweet_time
                #    self.calcGrowthRate()
                


                tweet_parts = tweet.strip().split(' ')
                try:
                    self.tweet_time  = int(tweet_parts[0])
                except ValueError:
                    print('Invalid document on line %d: %s' % (self.line, tweet))
                    continue
                
                self.tweet_time_notz = datetime.utcfromtimestamp(self.tweet_time * 0.001)
                tweet_time_utc = utc.localize(self.tweet_time_notz)
                
                if from_date is not None and tweet_time_utc < from_date:
                    continue
                    
                if to_date is not None and tweet_time_utc > to_date:
                    break
                    
                # TEMP ignore gameinsight spam and short tweets
                if len(tweet_parts) < 6 or tweet.find('gameinsight') != -1:
                    continue


                # allocate tweet to cluster
                doc_vec = document_to_vector(tweet_parts[2:], idfs)

                if doc_vec is None:
                    continue
                    
                lowest_index = self.lookupNearest(doc_vec)
                
                if lowest_index != -1:
                    c = self.clusters[lowest_index]
                    c.documents.append(doc_vec)
                    c.text_data.append([tweet.strip(), self.tweet_time])
                    c.last_update = self.line


                    # update the cluster center if the cluster is small
                    if len(c.documents) < 5:
                        self.lsh_engine.delete_vector(lowest_index)

                        c.center = np.mean(c.documents, axis=0)
                        c.norm   = np.linalg.norm(c.center)

                        self.lsh_engine.store_vector(c.center, lowest_index)
                    else:
                        if len(c.documents) < 100:
                            c.power = np.mean(np.std(c.documents, axis=0))
                else:
                    # no cluster found, construct new one
                    self.initNewCluster([doc_vec], [tweet.strip()], self.line, self.tweet_time, lang,[self.tweet_time])
            
        except KeyboardInterrupt:
            print("Line: %d Clusters: %d" % (self.line, len(self.clusters)))
            print("Cancelled")
 
 
def get_clusters():
    day = request.args.get('day')
    dt = parser.parse(day)
    if day not in analysers:
        analysers[day] = ClusterAnalyser()
        analysers[day].construct_clusters(opt_text, from_date=(dt-timedelta(hours=12)), to_date=(dt+timedelta(days=1)), idfs=idfs, lang=opt_lang)
        for c in analysers[day].clusters:
            analysers[day].clusters[c].calculateGrowthAndSentiment()
            analysers[day].clusters[c].analysis_day = day
    
    return simplejson.dumps(cluster_exporter.convert_to_dict(analysers[day].clusters,idfs, None, (dt - utc.localize(datetime(1970,1,1))).total_seconds()*1000))
    
cache = dict()
def get_cluster_data(cid):
    cid = int(cid)
    day = request.args.get('day')
    dt = parser.parse(day)
    if day not in analysers:
        return simplejson.dumps([])
    return cluster_exporter.collectDataForCluster(analysers[day].clusters[cid], cache)
        
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
    
    flask_app.add_url_rule('/cluster_data/cluster_<cid>.json', 'get_cluster_data', get_cluster_data) 
    flask_app.add_url_rule('/cluster_data_test.json', 'get_clusters', get_clusters) 
    flask_app.add_url_rule('/<path:path>', 'send_index', send_index)  
    flask_app.run(host='0.0.0.0',port='80', 
            debug = True, use_reloader=False)#, ssl_context=context

if __name__ == "__main__":
    main()
