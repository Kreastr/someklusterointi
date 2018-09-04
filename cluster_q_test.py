from clustering import get_cluster_data, ClusterAnalyser, idfs, opt_text, opt_lang
from datetime import datetime, timedelta
from pytz import utc
import time
import cluster_exporter
#cluster_exporter.collectDataForCluster
import json
import copy
scores =[]

parameters=[['entropyLikelyhood', [True, False], lambda x: True],
            ['single_cutter',[True, False], lambda x: x['entropyLikelyhood']],
            ['cutter_1',['getMinCut','getMinCutProb'], lambda x: (not x['single_cutter']) if x['entropyLikelyhood'] else False],
            ['cutter',['getMinCut','getMinCutProb','getClusterRandomKeywordSplit','getClusterEntropySplit','getRandomContractionsMinCut'], lambda x: ( x['single_cutter']) if x['entropyLikelyhood'] else False],
            ['cutter_2',['getClusterRandomKeywordSplit','getClusterEntropySplit','getRandomContractionsMinCut'], lambda x: (not x['single_cutter']) if x['entropyLikelyhood'] else False],
            ['cutter_threshold',[20,30,40,50], lambda x: (not x['single_cutter']) if x['entropyLikelyhood'] else False],
            ['usedropout', [True, False], lambda x: (  (x['cutter_1'] in ['getRandomContractionsMinCut','getMinCut','getMinCutProb']) or (x['cutter_2'] in ['getRandomContractionsMinCut','getMinCut','getMinCutProb'])) if 'cutter_1' in x else ((x['cutter'] in ['getRandomContractionsMinCut','getMinCut','getMinCutProb']) if 'cutter' in x else False)]]

def addParams(inlist, expansion):
    outlist = []
    for el in inlist:
        if expansion[2](el):
            for e in expansion[1]:
                tmp = copy.copy(el)
                tmp[expansion[0]] = e
                outlist.append(tmp)
        else:
            outlist.append(el)
    return outlist

def buildParams():
    outlist = [dict(overrun=False)]
    for p in parameters:
        outlist = addParams(outlist, p)
    return outlist
    
def run():
    params = json.load(open('valid-par.json'))#buildParams()
    print (params)
    init_time = time.time()
    for par in params:
        par['overrun'] = False
    for trials in range(500):
        i = 0
        print (i)
        for par in params:
            if par['overrun']:
                continue
            i += 1
            analyser = ClusterAnalyser()
            analyser.max_runtime = 3600*4
            analyser.min_lines_per_second = 0
            
            analyser.cutter = []
            for n,p in par.iteritems():
                if n == 'usedropout':
                    analyser.simgraphparams['usedropout'] = p
                elif n == 'cutter':
                    analyser.cutter.append([0, None, p])
                elif n == 'cutter_1':
                    analyser.cutter.append([0, par['cutter_threshold'], p])
                elif n == 'cutter_2':
                    analyser.cutter.append([par['cutter_threshold'], None, p])
                elif n == 'cutter_threshold':
                    pass
                elif n == 'entropyLikelyhood':
                    analyser.entropyLikelyhood = p
            
            dt = utc.localize(datetime(2014, 7, 3))
            start_time = time.time()
            analyser.construct_clusters(opt_text, from_date=(dt+timedelta(hours=0)), to_date=(dt+timedelta(hours=24)), idfs=idfs, lang=opt_lang)
            time_elapsed = time.time()-start_time
            if time_elapsed > analyser.max_runtime or analyser.overrun:
                par['overrun'] = True
                continue
            summ = 0.0
            weights = 0.0

            for cid, cluster in analyser.clusters.iteritems():
                e = cluster_exporter.calculate_cluster_entropy(cluster, idfs)
                summ += e*len(cluster.documents)
                weights += len(cluster.documents)
                
            total_elapsed = time.time()-init_time
            scores.append([par, summ/weights, len(analyser.clusters), time_elapsed])
            print (par)
            print ('%d-%d: Weighted entropy: %f Cluster count: %d, Total elapsed: %d'  % (i, trials, summ/weights, len(analyser.clusters), total_elapsed))
            json.dump(scores,open('run-res.json.bak','w'))
            json.dump(scores,open('run-res.json','w'))
    
run()
    
