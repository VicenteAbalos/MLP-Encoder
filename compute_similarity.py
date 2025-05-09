import numpy as np
import torch
import skimage.io as io
import skimage.transform as transform
import matplotlib.pyplot as plt
import os

# load the data for visualizing the results
data_dir = 'simple1K/simple1K/'
image_dir = os.path.join(data_dir, 'images')
val_file = os.path.join(data_dir, 'list_of_images.txt')
#

#funcion del precision
def pr1(idx):
    p=0
    list_p=[]
    correct_c=0
    for i in range(1,len(idx)):
        if idx[i]==1:
            correct_c+=1
            p+=correct_c/i
            list_p.append(correct_c/i)
        #print(idx[i],correct_c,p)
    if correct_c>0:
        p=p/correct_c
    return p, list_p
#
#Recall maker: returned dictionary must be saved in another dictionary
def recall_vect(l_p):
    recall={}
    total=len(l_p)
    currentp=0
    i=0
    while i<11:
        if i/10<=(currentp+1)/total:
            recall[i/10]=l_p[currentp]
            i+=1
        else:
            currentp+=1
    return recall
#
#Recall avg
def mean_recall(recall):
    vector=[0,0,0,0,0,0,0,0,0,0,0] #recall vector where I'll put every average
    for position in range(11):
        for row in recall.keys():
            vector[position]+=recall[row][position/10] #this should add all values of the same position to the corresponding vector value
        vector[position]=vector[position]/len(recall) #this should bring the value down to its average
    return vector
#

DATASET = 'simple1k'
MODEL = 'resnet34'
MODEL2 = 'resnet18'
DINOV2 = 'dinov2'
CLIP = 'clip'

feat_file = os.path.join('data', 'feat_{}_{}.npy'.format(MODEL, DATASET))
if __name__ == '__main__' :
    ll=[]
    for i in range(1):
        with open(val_file, "r+") as file: 
            files = [f.split('\t') for f in file]
        #--- compute similarity
        feats = np.load(feat_file)    
        norm2 = np.linalg.norm(feats, ord = 2, axis = 1,  keepdims = True)
        feats_n = feats / norm2
        sim = feats_n @ np.transpose(feats_n)
        sim_idx = np.argsort(-sim, axis = 1)
        #sime_idx = np.argsort(sim,axis=1)
        
        #---- An example of results just pickin a random query
        # the first image appearing must be the same as the query
        query = np.random.permutation(sim.shape[0])[0]
        k = 10
        best_idx = sim_idx[query, :k+1]
        #worst_idx = sime_idx[query, :k+1]
        print(sim_idx[query])
        print(sim[query, best_idx])
        print("row 1:",sim[0])

        mAP=0
        AP_list=[]
        R_dict={}
        for j, row in enumerate(sim_idx):
            val_list=[]
            for i, idx in enumerate(row):        
                #print("i:",i,"idx:",idx)
                #filename = os.path.join(image_dir, files[idx][0])
                if files[idx][1]==files[row[0]][1]:
                    val_list.append(1)
                else:
                    val_list.append(0)
            #print("Val:",val_list)
            avg_precision,precision_list=pr1(val_list)
            R_dict[j]=recall_vect(precision_list) ###This should save the recall vector in a dictionary with an idx key
            AP_list.append(avg_precision)
            mAP+=avg_precision
        mAP=mAP/len(AP_list)
        print(mAP)
        #print("recall dict:",R_dict)
        recall_avg=mean_recall(R_dict) ###This should return the average vector of recalls
        #print("avg recall:",recall_avg)

        best_list=AP_list.copy()
        best_list.sort()
        best_five=best_list[-5:]
        worst_five=best_list[:5]
        print("best list:",AP_list.index(best_list[-100]))

        index=AP_list.index(best_five[0])
        iter_best=0
        index_best=[]
        while len(index_best)<5:
            for val in AP_list:
                if val==best_five[iter_best] and AP_list.index(val) not in index_best:
                    print("yay")
                    index_best.append(AP_list.index(val))
                    iter_best+=1
                    break
                elif val==best_five[iter_best]:
                    index_best.append(AP_list.index(val)+iter_best)
                    iter_best+=1
                    break
        print("WE DID IT:",index_best)

        the_best_idx = sim_idx[index, :k+1]
        index2=AP_list.index(worst_five[4])
        the_worst_idx = sim_idx[index2, :k+1]

        fig, ax = plt.subplots(5,11)
        w = 0
        #val_list=[]
        for j in range(len(best_five)):
            #index=AP_list.index(best_five[j])
            index=index_best[j]
            #print(index)
            the_best_idx=sim_idx[index,:11]
            for i, idx in enumerate(the_best_idx):        
                #print("i:",i,"idx:",idx)
                filename = os.path.join(image_dir, files[idx][0])
                im = io.imread(filename)
                im = transform.resize(im, (64,64)) 
                ax[j,i].imshow(im)                 
                ax[j,i].set_axis_off()
                ax[j,i].set_title(files[idx][1])
                """if files[idx][1]==files[query][1]:
                    val_list.append(1)
                else:
                    val_list.append(0)"""
            #print(sim[query,best_idx][1:])
            #print(val_list)
            #print(pr1(val_list))
        
        ax[0,0].patch.set(lw=6, ec='b')
        ax[0,0].set_axis_on()            
        plt.show()