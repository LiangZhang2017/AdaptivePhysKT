
import numpy as np

def corcount(df,index):
    temp=np.repeat(0,[len(df['CF_ansbin'])],axis=0)
    for i in np.unique(index):
        corAll = list(np.cumsum(df['CF_ansbin'][index == i] == 1))
        rangeIndex=list(range(0,len(np.cumsum(df['CF_ansbin'][index==i]))-1))

        extract=[corAll[i] for i in rangeIndex]
        extract.insert(0,0)
        temp[index==i]=extract
    return temp

def incorcount(df,index):
    temp=np.repeat(0,[len(df['CF_ansbin'])],axis=0)
    for i in np.unique(index):
        corAll = list(np.cumsum(df['CF_ansbin'][index == i] == 0))
        rangeIndex=list(range(0,len(np.cumsum(df['CF_ansbin'][index==i]))-1))

        extract=[corAll[i] for i in rangeIndex]
        extract.insert(0,0)
        temp[index==i]=extract
    return temp

def studycount(df,index):
    temp=np.repeat(0,[len(df['CF_ansbin'])],axis=0)
    for i in np.unique(index):
        corAll = list(np.cumsum(df['CF_ansbin'][index == i] == -1))
        rangeIndex=list(range(0,len(np.cumsum(df['CF_ansbin'][index==i]))-1))

        extract=[corAll[i] for i in rangeIndex]
        extract.insert(0,0)
        temp[index==i]=extract
    return temp

def CheckKCMatch(train,test,kc_used):
    keep_theKC_used = [kc for kc in np.unique(train[kc_used]) if kc in np.unique(test[kc_used])]
    test_new=test.loc[test[kc_used].isin(keep_theKC_used)].copy()

    return train,test_new