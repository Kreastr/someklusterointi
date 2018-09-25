import json
res0 = json.load(open('/home/aromanenko/Documents/results_l0.json'))
res = json.load(open('/home/aromanenko/Documents/results_l2.json'))+res0
import numpy as np
import pandas as pd
from scipy import stats, integrate
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kstest, norm, linregress
n_first = 40

def plotDF(transform, nrows, ncols, fltr, separated, figsize, lbls, filename, plotter, axLabels=None):
    f, ax = plt.subplots(figsize=figsize, ncols=ncols,nrows=nrows)
    r = [list(map(transform,separated[i][1])) for i in range(len(separated)) if fltr(separated[i][1][0][0])]
    df = [pd.DataFrame(r[i], columns=["x", "y"]) for i in range(len(r))]
    ymin = min([min(x.y) for x in df])
    ymax = max([max(x.y) for x in df])
    xmin = min([min(x.x) for x in df])
    xmax = max([max(x.x) for x in df])
    titles = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)','(k)']
    for i in range(len(df)):
        if nrows > 1 and ncols > 1:
            a = ax[i % numrows, int(i/numrows)]
        elif nrows > 1 or ncols > 1:
            a = ax[i]
        else:
            a = ax
        plotter(df[i].x, df[i].y, ax=a,label=(lbls[i] if lbls else None)) 
        a.set_xlim(xmin, xmax)
        a.set_ylim(ymin, ymax)
        if not axLabels is None:
            a.set(**axLabels)
        a.legend(loc="best")
        if len(df) > 1:
            a.set_title(titles[i])
        
    f.tight_layout()
    plt.savefig(filename)



r = list(map(lambda x: [x[1], x[2]],res))
uniq=list(set(list(map(lambda x: hash(str(x[0])),res))))
separated = [(u, list(filter(lambda x: hash(str(x[0])) == u,res))[:n_first]) for u in uniq]
separated.sort(key=lambda x: (0 if x[1][0][0]['single_cutter'] else x[1][0][0]['cutter_threshold']) if 'single_cutter' in x[1][0][0] else -1)


def normalize(x):
    x1 = np.array(x)
    return (x1 - np.mean(x1))/np.std(x1)

fltr = lambda x: (not x['single_cutter'] if 'single_cutter' in x else False) and (not x['cutter_1'] == 'getMinCut') and not x['usedropout'] and x['cutter_2'] == 'getRandomContractionsMinCut'

r2 = '\n----------------\n'.join(['\n'.join([str(separated[i][1][0][0]),str(np.mean(list(map(lambda x: x[3],separated[i][1])))),str(np.mean(list(map(lambda x: x[1]*x[2],separated[i][1])))),str(np.std(list(map(lambda x: x[1]*x[2],separated[i][1]))))]) for i in range(len(separated)) if fltr(separated[i][1][0][0])])
print (r2)

cutter_index = dict(getMinCut="B",
                    getMinCutProb="C",
                    getClusterRandomKeywordSplit="G",
                    getClusterEntropySplit="F",
                    getRandomContractionsMinCut="D")    

r3 = [['%s%s%s%s%s%s'%('T' if not separated[i][1][0][0]['entropyLikelyhood'] else 'F', 
                                   '-' if not separated[i][1][0][0]['entropyLikelyhood'] else ('1' if separated[i][1][0][0]['single_cutter'] else '2'),
                                   '-' if not separated[i][1][0][0]['entropyLikelyhood'] else (cutter_index[separated[i][1][0][0]['cutter']] if separated[i][1][0][0]['single_cutter'] else cutter_index[separated[i][1][0][0]['cutter_1']]) ,
                                   '-' if not separated[i][1][0][0]['entropyLikelyhood'] else ('-' if separated[i][1][0][0]['single_cutter'] else cutter_index[separated[i][1][0][0]['cutter_2']]),
                                   '-' if not separated[i][1][0][0]['entropyLikelyhood'] else ('--' if separated[i][1][0][0]['single_cutter'] else str(separated[i][1][0][0]['cutter_threshold'])),
                                   '-' if not separated[i][1][0][0]['entropyLikelyhood'] else (('D' if separated[i][1][0][0]['usedropout'] else '-') if 'usedropout' in separated[i][1][0][0] else '-'),),
                                   '%.1f' % (np.mean(list(map(lambda x: x[3],separated[i][1])))),
                                   '%.1f' % (np.mean(list(map(lambda x: x[1]*x[2],separated[i][1])))),
                                   '%.1f' % (np.std(list(map(lambda x: x[1]*x[2],separated[i][1]))))
                                   ] for i in range(len(separated)) ]

r3.sort(key=lambda x: x[2])
print ('\\\\\n'.join(map(lambda x: '\t & \t'.join(x),r3)))


r4 = [['%s%s%s%s%s%s'%('T' if not separated[i][1][0][0]['entropyLikelyhood'] else 'F', 
                                   '-' if not separated[i][1][0][0]['entropyLikelyhood'] else ('1' if separated[i][1][0][0]['single_cutter'] else '2'),
                                   '-' if not separated[i][1][0][0]['entropyLikelyhood'] else (cutter_index[separated[i][1][0][0]['cutter']] if separated[i][1][0][0]['single_cutter'] else cutter_index[separated[i][1][0][0]['cutter_1']]) ,
                                   '-' if not separated[i][1][0][0]['entropyLikelyhood'] else ('-' if separated[i][1][0][0]['single_cutter'] else cutter_index[separated[i][1][0][0]['cutter_2']]),
                                   '-' if not separated[i][1][0][0]['entropyLikelyhood'] else ('--' if separated[i][1][0][0]['single_cutter'] else str(separated[i][1][0][0]['cutter_threshold'])),
                                   '-' if not separated[i][1][0][0]['entropyLikelyhood'] else (('D' if separated[i][1][0][0]['usedropout'] else '-') if 'usedropout' in separated[i][1][0][0] else '-'),),
                                   '%2.1f' % (100*kstest(normalize(list(map(lambda x: x[1],separated[i][1]))), norm.cdf)[1]),
                                   '%2.1f' % (100*kstest(normalize(list(map(lambda x: x[2],separated[i][1]))), norm.cdf)[1]),
                                   '%2.1f' % (100*kstest(normalize(list(map(lambda x: x[1]*x[2],separated[i][1]))), norm.cdf)[1]),
                                   ] for i in range(len(separated)) ]

r4.sort(key=lambda x: x[0])
print ('\\\\\n'.join(map(lambda x: '\t & \t'.join(x),r4)))

r5 = [['%s%s%s%s%s%s'%('T' if not separated[i][1][0][0]['entropyLikelyhood'] else 'F', 
                                   '-' if not separated[i][1][0][0]['entropyLikelyhood'] else ('1' if separated[i][1][0][0]['single_cutter'] else '2'),
                                   '-' if not separated[i][1][0][0]['entropyLikelyhood'] else (cutter_index[separated[i][1][0][0]['cutter']] if separated[i][1][0][0]['single_cutter'] else cutter_index[separated[i][1][0][0]['cutter_1']]) ,
                                   '-' if not separated[i][1][0][0]['entropyLikelyhood'] else ('-' if separated[i][1][0][0]['single_cutter'] else cutter_index[separated[i][1][0][0]['cutter_2']]),
                                   '-' if not separated[i][1][0][0]['entropyLikelyhood'] else ('--' if separated[i][1][0][0]['single_cutter'] else str(separated[i][1][0][0]['cutter_threshold'])),
                                   '-' if not separated[i][1][0][0]['entropyLikelyhood'] else (('D' if separated[i][1][0][0]['usedropout'] else '-') if 'usedropout' in separated[i][1][0][0] else '-'),),
                                   '%d' % (len(separated[i][1])),
                                   ] for i in range(len(separated)) ]

r5.sort(key=lambda x: x[0])
print ('\\\\\n'.join(map(lambda x: '\t & \t'.join(x),r5)))

print ('------------F2BFxxD--------------------')
#F2BFxxD
ncols=1
nrows=1
plotDF(transform=lambda x: [x[1]*x[2],x[3]], 
       nrows=nrows, 
       ncols=ncols, 
       fltr = lambda x: (not x['single_cutter'] and (x['cutter_1'] == 'getMinCut') and x['usedropout'] and x['cutter_2'] == 'getClusterEntropySplit')  if x['entropyLikelyhood'] else False, 
       separated=separated, 
       figsize=(3*ncols,3*nrows), 
       lbls=['30','40','50'], 
       filename='perfomance_vs_time_F2BFxxD.png',
       plotter = sns.regplot,
       axLabels=dict(xlabel="Performance score",ylabel="Computation time, s"))#kdeplot

fltr = lambda x: (not x['single_cutter'] and (x['cutter_1'] == 'getMinCut') and x['usedropout'] and x['cutter_2'] == 'getClusterEntropySplit')  if x['entropyLikelyhood'] else False;
([print((linregress([[x[1], x[2]] for x in separated[i][1]]), separated[i][1][0][0]['cutter_threshold'])) for i in range(len(separated)) if fltr(separated[i][1][0][0])])



#F2CFxxD
print ('------------F2CFxxD--------------------')
ncols=1
nrows=1
plotDF(transform=lambda x: [x[1]*x[2],x[3]], 
       nrows=nrows, 
       ncols=ncols, 
       fltr = lambda x: (not x['single_cutter'] and (x['cutter_1'] == 'getMinCutProb') and x['usedropout'] and x['cutter_2'] == 'getClusterEntropySplit')  if x['entropyLikelyhood'] else False, 
       separated=separated, 
       figsize=(3*ncols,3*nrows), 
       lbls=['20','40','50'], 
       filename='perfomance_vs_time_F2CFxxD.png',
       plotter = sns.regplot,
       axLabels=dict(xlabel="Performance score",ylabel="Computation time, s"))#kdeplot

fltr = lambda x: (not x['single_cutter'] and (x['cutter_1'] == 'getMinCutProb') and x['usedropout'] and x['cutter_2'] == 'getClusterEntropySplit')  if x['entropyLikelyhood'] else False;
([print((linregress([[x[1], x[2]] for x in separated[i][1]]), separated[i][1][0][0]['cutter_threshold'])) for i in range(len(separated)) if fltr(separated[i][1][0][0])])

print ('------------F2CDxxD--------------------')
#F2CDxxD
ncols=1
nrows=1
plotDF(transform=lambda x: [x[1]*x[2],x[3]], 
       nrows=nrows, 
       ncols=ncols, 
       fltr = lambda x: (not x['single_cutter'] and (x['cutter_1'] == 'getMinCutProb') and x['usedropout'] and x['cutter_2'] == 'getRandomContractionsMinCut')  if x['entropyLikelyhood'] else False, 
       separated=separated, 
       figsize=(3*ncols,3*nrows), 
       lbls=['30','40','50'], 
       filename='perfomance_vs_time_F2CDxxD.png',
       plotter = sns.regplot,
       axLabels=dict(xlabel="Performance score",ylabel="Computation time, s"))#kdeplot

fltr = lambda x: (not x['single_cutter'] and (x['cutter_1'] == 'getMinCutProb') and x['usedropout'] and x['cutter_2'] == 'getRandomContractionsMinCut')  if x['entropyLikelyhood'] else False;
([print((linregress([[x[1], x[2]] for x in separated[i][1]]), separated[i][1][0][0]['cutter_threshold'])) for i in range(len(separated)) if fltr(separated[i][1][0][0])])


ncols=1
nrows=3
plotDF(transform=lambda x: [x[1], x[2]], 
       nrows=nrows, 
       ncols=ncols, 
       fltr= lambda x: x['single_cutter'] if 'single_cutter' in x else False, 
       separated=separated, 
       figsize=(3*ncols,3*nrows), 
       lbls=None, 
       filename='paret_density_single_cutter.png',
       plotter = sns.kdeplot,
       axLabels=dict(xlabel="Average entropy, bit",ylabel="Number of clusters"))

ncols=1
nrows=3
plotDF(transform=lambda x: [x[1]*x[2],x[3]], 
       nrows=nrows, 
       ncols=ncols, 
       fltr= lambda x: x['single_cutter'] if 'single_cutter' in x else False, 
       separated=separated, 
       figsize=(3*ncols,3*nrows), 
       lbls=None, 
       filename='perfomance_vs_time_single_cutter.png',
       plotter = sns.kdeplot,
       axLabels=dict(xlabel="Performance score",ylabel="Computation time, s"))#kdeplot



ncols=1
nrows=3
plotDF(transform=lambda x: [x[1], x[2]], 
       nrows=nrows, 
       ncols=ncols, 
       fltr = lambda x: (not x['single_cutter'] and (x['cutter_1'] == 'getMinCutProb')  and x['cutter_2'] == 'getRandomContractionsMinCut'and x['usedropout'])  if x['entropyLikelyhood'] else False, 
       separated=separated, 
       figsize=(3*ncols,3*nrows), 
       lbls=None, 
       filename='paret_density_getMinCutProb_getRandomContractionsMinCut_usedropout.png',
       plotter = sns.kdeplot,
       axLabels=dict(xlabel="Average entropy, bit",ylabel="Number of clusters"))

ncols=1
nrows=1
plotDF(transform=lambda x: [x[1]*x[2],x[3]], 
       nrows=nrows, 
       ncols=ncols, 
       fltr = lambda x: (not x['single_cutter'] and (x['cutter_1'] == 'getMinCutProb')  and x['cutter_2'] == 'getRandomContractionsMinCut'and x['usedropout'])  if x['entropyLikelyhood'] else False, 
       separated=separated, 
       figsize=(3*ncols,3*nrows), 
       lbls=None, 
       filename='perfomance_vs_time_getMinCutProb_getRandomContractionsMinCut_usedropout.png',
       plotter = sns.regplot,
       axLabels=dict(xlabel="Performance score",ylabel="Computation time, s"))#kdeplot


ncols=1
nrows=3
plotDF(transform=lambda x: [x[1], x[2]], 
       nrows=nrows, 
       ncols=ncols, 
       fltr= lambda x: (not x['single_cutter'] and (x['cutter_1'] == 'getMinCut') and x['usedropout'] and x['cutter_2'] == 'getRandomContractionsMinCut')  if x['entropyLikelyhood'] else False, 
       separated=separated, 
       figsize=(3*ncols,3*nrows), 
       lbls=None, 
       filename='paret_density_getMinCut_getRandomContractionsMinCut_usedropout.png',
       plotter = sns.kdeplot,
       axLabels=dict(xlabel="Average entropy, bit",ylabel="Number of clusters"))

ncols=1
nrows=1
plotDF(transform=lambda x: [x[1]*x[2],x[3]], 
       nrows=nrows, 
       ncols=ncols, 
       fltr= lambda x: (not x['single_cutter'] and (x['cutter_1'] == 'getMinCut') and x['usedropout'] and x['cutter_2'] == 'getRandomContractionsMinCut')  if x['entropyLikelyhood'] else False, 
       separated=separated, 
       figsize=(3*ncols,3*nrows), 
       lbls=['30','40','50'], 
       filename='perfomance_vs_time_getMinCut_getRandomContractionsMinCut_usedropout.png',
       plotter = sns.regplot,
       axLabels=dict(xlabel="Performance score",ylabel="Computation time, s"))#kdeplot

ncols=1
nrows=3
plotDF(transform=lambda x: [x[1], x[2]], 
       nrows=nrows, 
       ncols=ncols, 
       fltr= lambda x: (not x['single_cutter'] and (x['cutter_1'] == 'getMinCutProb') and x['usedropout'] and x['cutter_2'] == 'getRandomContractionsMinCut')  if x['entropyLikelyhood'] else False, 
       separated=separated, 
       figsize=(3*ncols,3*nrows), 
       lbls=None, 
       filename='paret_density_getMinCutProb_getRandomContractionsMinCut_usedropout.png',
       plotter = sns.kdeplot,
       axLabels=dict(xlabel="Average entropy, bit",ylabel="Number of clusters"))

ncols=1
nrows=1
plotDF(transform=lambda x: [x[1]*x[2],x[3]], 
       nrows=nrows, 
       ncols=ncols, 
       fltr= lambda x: (not x['single_cutter'] and (x['cutter_1'] == 'getMinCutProb') and x['usedropout'] and x['cutter_2'] == 'getRandomContractionsMinCut')  if x['entropyLikelyhood'] else False, 
       separated=separated, 
       figsize=(3*ncols,3*nrows), 
       lbls=['30','40','50'], 
       filename='perfomance_vs_time_getMinCutProb_getRandomContractionsMinCut_usedropout.png',
       plotter = sns.regplot,
       axLabels=dict(xlabel="Performance score",ylabel="Computation time, s"))#kdeplot

ncols=1
nrows=3
plotDF(transform=lambda x: [x[1], x[2]], 
       nrows=nrows, 
       ncols=ncols, 
       fltr= lambda x:(not x['single_cutter'] and (not x['cutter_1'] == 'getMinCut') and x['usedropout'] and x['cutter_2'] == 'getRandomContractionsMinCut')  if x['entropyLikelyhood'] else False, 
       separated=separated, 
       figsize=(3*ncols,3*nrows), 
       lbls=None, 
       filename='paret_density_getMinCut_usedropout.png',
       plotter = sns.kdeplot,
       axLabels=dict(xlabel="Average entropy, bit",ylabel="Number of clusters"))

#ncols=1
#nrows=1
#plotDF(transform=lambda x: [x[1]*x[2],x[3]], 
       #nrows=nrows, 
       #ncols=ncols, 
       #fltr= lambda x: (not x['single_cutter'] and (not x['cutter_1'] == 'getMinCut') and x['usedropout'] and x['cutter_2'] == 'getRandomContractionsMinCut')  if x['entropyLikelyhood'] else False, 
       #separated=separated, 
       #figsize=(3*ncols,3*nrows), 
       #lbls=['30','40','50'], 
       #filename='perfomance_vs_time_getMinCut_getRandomContractionsMinCut_usedropout.png',
       #plotter = sns.regplot,
       #axLabels=dict(xlabel="Performance score",ylabel="Computation time, s"))#kdeplot


ncols=2
nrows=1
plotDF(transform=lambda x: [x[1], x[2]], 
       nrows=nrows, 
       ncols=ncols, 
       fltr= lambda x: (not x['single_cutter'] and (not x['cutter_1'] == 'getMinCut') and not x['usedropout'] and x['cutter_2'] == 'getRandomContractionsMinCut')  if x['entropyLikelyhood'] else False, 
       separated=separated, 
       figsize=(3*ncols,3*nrows), 
       lbls=None, 
       filename='paret_density_getMinCut_getRandomContractionsMinCut_nodropout.png',
       plotter = sns.kdeplot,
       axLabels=dict(xlabel="Average entropy, bit",ylabel="Number of clusters"))

ncols=1
nrows=1
plotDF(transform=lambda x: [x[1]*x[2],x[3]], 
       nrows=nrows, 
       ncols=ncols, 
       fltr= lambda x: (not x['single_cutter'] and (not x['cutter_1'] == 'getMinCut') and not x['usedropout'] and x['cutter_2'] == 'getRandomContractionsMinCut')  if x['entropyLikelyhood'] else False, 
       separated=separated, 
       figsize=(3*ncols,3*nrows), 
       lbls=['30','40','50'], 
       filename='perfomance_vs_time_getMinCut_getRandomContractionsMinCut_nodropout.png',
       plotter = sns.regplot,
       axLabels=dict(xlabel="Performance score",ylabel="Computation time, s"))#kdeplot


