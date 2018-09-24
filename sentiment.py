import pickle 
import numpy as np

lsvc = pickle.load(open("sentiment.clf","rb"))

def calcRating(df):
    return (df[0]-df[-1])/np.sum(df)
        
def getSentiment(mdv):
    return calcRating(lsvc.predict_proba([mdv])[0])#/(max(1.0,20.-len(wvecs)))
