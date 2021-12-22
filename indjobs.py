#place to put utility functions
import pandas as pd
import numpy as np

def get_confusion(CF, i):
    """
    demonstration of taking apart the multi-variate confusion matrix to get the 2x2
    and also calculate the common metrics for each classification group
    CF: a multivariate confusiotn matrix (DataFrame from ndarray)
    i: the variable index for which to show the traditional 2x2 confusion matrix
    
    computed accuracy score matches with the 'macro avg' computed by sklearn.metrics.classification_report
    """
    TP = CF.loc[i,i]
    FN = CF.loc[i].sum() - TP #the row (predictions)
    FP = CF.loc[:, i].sum() - TP #the column (actuals)
    TN = CF.values.sum() - CF.loc[i].sum() - CF.loc[:,i].sum() + TP # or CF.values.sum() - FN - FP
    #since TP is subtracted twice (as part of the row and as part of the column - need to add it back once to get the right total)
    cfout =pd.DataFrame([[TP,FN],[FP,TN]], columns = ['pred_pos', 'pred_neg'], index=['act_pos', 'act_neg'])
    #act_neg = FP + TN
    accuracy = (TP+TN)/(TP+TN+FP+FN) # (TP+TN)/(TP+FN)
    precision = TP / (TP+FP) #yes, but no? #TP/pred_pos #pred_pos = TP + FP # 
    recall =  TP/(TP+FN) #TP/act_pos #act_pos = TP + FN#
    f1 = 2*(precision * recall) / (precision+recall)
    #print((i), f"TP is {TP}; FP is {FP}; FN is {FN}; TN is {TN}" )
    cols = ['TP', 'FP', 'FN', 'TN', 'accuracy', 'precision', 'recall', 'f1']
    res = pd.Series([TP, FP, FN, TN, accuracy, precision, recall, f1 ], index=cols)
    return res

def eudist(p1, p2):
    #sqrt of (x2-x1)**2 + (y2-y1)**2
    return np.sqrt((p2[0]-p1[0])**2+(p2[1]-p1[1])**2)

def npdist(p1, p2):
    """calculate distance using numpy.linalg.norm
     Euclidean distance is equivalent to the l2 norm of the difference between the two points
     i.e. np.linalg.norm(np.array((10,0))-np.array((1,0)),2) == 9
    """
    return np.linalg.norm(p2-p1, 2)

def test_eudist():
    
    p1 = (0, 0)
    p2 = (0, 0)
    assert eudist(p1,p2) == 0
    
    p2 = (10,0)
    assert eudist(p1,p2) == 10
    
    p2 = (0, 10)
    assert eudist(p1,p2) ==10
    
    p2 = (4, 3)
    assert eudist(p1,p2) ==5

def inersha(pts, ctrs): 
    """ 
    sum of squared distances of points from their nearest centroid
    given points and centroids - compute the inertia (how close each point is to nearest centrooid)
    return the sum
    """
    res = []
    for pt in pts:
        #compute distances from this point to each center an keep the minimum one
        print(f"for point {pt}", end="---")
        mindist = np.inf
        for c in ctrs:
            #commpute distance
            dist = eudist(pt, c)
            if dist < mindist: 
                msg = f"{c} is closest with dist {dist:.2f}"
                mindist = dist
        print(msg)
        res.append(mindist**2)
    return sum(res)

def test_inersha():
    pts = [(0,10),(5,5),(4,3),(1,1),(3,4), (8, 5), (0, 1), (5, 1), (5, 9), (0, 8), (1, 5), (1, 8), (1, 3), (2, 6), (6, 2), [1, 2], [8, 2], [4, 0], [6, 2], [4, 9], [8, 6], [9, 2], [3, 2], [7, 5], [6, 5]]
    ctrs = [(0,0), (0,10), (7,7)]
    inersha(pts, ctrs) == 330
    
def get_all_confusion(CF, target_names):
    allgroups = pd.Series(CF.index).apply(lambda x: get_confusion(CF, x))
    avgofrows = allgroups.mean()
    avgofrows.name='ROW AVG'
    allgroups.index = target_names #replace numeric index with category names
    return pd.concat((allgroups, avgofrows.to_frame().T))

def get_newsgroup_data(uselocalcopy = True, remove = ('headers', 'footers', 'quotes'), categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']):
    import shelve as s
    import os
    shelfloc = r'c:\temp\20newsgroups.dat'
    data_train = None
    data_test = None
    if os.path.exists(shelfloc) and uselocalcopy:
        with s.open(os.path.splitext(shelfloc)[0]) as datastore:
            try: 
                data_train = datastore['data_train']
                data_test = datastore['data_test']
            except KeyError as e:
                print(f'ERROR: could not find key: {e.args[0].decode()}')
                get_newsgroup_data(False)
    else:
        print(f'updating local copy at {shelfloc}')
        from sklearn.datasets import fetch_20newsgroups
        data_train = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42, remove=remove)
        data_test = fetch_20newsgroups(subset="test", categories=categories, shuffle=True, random_state=42, remove=remove)
        
        #save for later use without doing download
        with s.open(os.path.splitext(shelfloc)[0]) as datastore:
            datastore['data_train'] = data_train
            datastore['data_test'] = data_test
    return data_train, data_test

def size_mb(docs):
    return sum(len(s.encode("utf-8")) for s in docs) / 1e6