import gzip
import random
import cPickle as pickle

class Cluster:
    def __init__(self):
        self.center    = None
        self.norm      = None
        self.documents = []
        self.text      = []
        self.last_update = 0


cluster_tweet_counts = []
cluster_deltas = []
cluster_delta_deltas = []
clusters = []

def load_clusters(filename):
    global clusters
    with gzip.open(filename) as f:
        clusters = pickle.load(f)

def calculate_sizes():
    for i in range(1, 8):
        filename = 'clusters_ru_tweets_2013-2015_lem_alpha_true-ru.txt_%d.bin.gz' % (i * 1000000)
        print('Reading file: %s' % filename)
        with gzip.open(filename, 'rb') as f:
            cluster_snapshot = pickle.load(f)
            print('File loaded')
            print('Counting cluster sizes...')

            cluster_counts = {}

            # Using first tweet in cluster as identifier as no unique ids are available
            for c in cluster_snapshot:
                cluster_counts[c.text[0]] = len(c.text)

            cluster_tweet_counts.append(cluster_counts)

def calculate_deltas(input, output):
    for i in range(len(input) - 1):
        deltas = {}
        for id, count in input[i].iteritems():
            next_count = input[i + 1].get(id, 0)
            deltas[id] = next_count - count

        output.append(deltas)

def print_trending_clusters():
    for cluster_delta in cluster_delta_deltas:
        for c in sorted(cluster_delta, key=cluster_delta.get)[-5:]:
            print('\n\n\n' + str(cluster_delta[c]) + ' -------------')
            for cluster in clusters:
                if cluster.text[0] == c:
                    if len(cluster.text) < 10:
                        print("LESS THEN TEN TWEETS IN CLUSTER, ERROR?")
                    else:
                        print('\n'.join(random.sample(cluster.text, 10)))
    
        print('################')

calculate_sizes()
calculate_deltas(cluster_tweet_counts, cluster_deltas)
calculate_deltas(cluster_deltas, cluster_delta_deltas)
load_clusters("clusters_ru_tweets_2013-2015_lem_alpha_true-ru.txt_4000000.bin.gz")
print_trending_clusters()
