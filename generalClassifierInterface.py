import pickle 
import numpy as np

def getMDV(vec):
    #print (vec)
    if len(vec) == 0:
        return [1]
    tmp = np.zeros((len(vec[0]),))
    for v in vec:
        tmp += v
    tmp /= len(vec)
    return tmp
    
class generalClassifierInterface():
    
    def __init__(self, dictionary, classifier, class_tags=None):
        self.bc = pickle.load(open(dictionary,"rb"))
        self.clf = pickle.load(open(classifier,"rb"))
        if class_tags:
            self.ct = pickle.load(open(class_tags,"rb"))
        else:
            self.ct = None
            
    def getTags(self, messages, top_n=1):
        wvecs = []
        for m in messages:
            for w in m:
                if w in self.bc:
                    wvecs.append(self.bc[w])
        
        if len(wvecs) > 0:
            mdv = getMDV(wvecs)
            classid = self.clf.predict_proba([mdv])[0]
            #print (classid)
            best_n = np.argsort(classid)[-top_n:]
            #print (best_n)
            if self.ct:
                return [self.ct[classid] for classid in best_n][::-1]
            else:
                return best_n
        else: 
            return None

 
