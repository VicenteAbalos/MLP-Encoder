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
    correct_c=0
    for i in range(1,len(idx)):
        if idx[i]==1:
            correct_c+=1
            p+=correct_c/i
        #print(idx[i],correct_c,p)
    if correct_c>0:
        p=p/correct_c
    return p
#

DATASET = 'simple1k'
MODEL = 'resnet34'
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
            avg_precision=pr1(val_list)
            AP_list.append(avg_precision)
            mAP+=avg_precision
        mAP=mAP/len(AP_list)
        print(mAP)

        """mAP=0
        for row in sim:
            ap=-1
            for i in row:
                ap+=i
            ap=ap/(len(row)-1)
            #print("Avg Precision:",ap)
            mAP+=ap
        mAP=mAP/len(sim)
        print("mAP:",mAP)"""

        fig, ax = plt.subplots(1,11)
        w = 0
        val_list=[]
        for i, idx in enumerate(best_idx):        
            #print("i:",i,"idx:",idx)
            filename = os.path.join(image_dir, files[idx][0])
            im = io.imread(filename)
            im = transform.resize(im, (64,64)) 
            ax[i].imshow(im)                 
            ax[i].set_axis_off()
            ax[i].set_title(files[idx][1])
            if files[idx][1]==files[query][1]:
                val_list.append(1)
            else:
                val_list.append(0)
        print(sim[query,best_idx][1:])
        print(val_list)
        print(pr1(val_list))
        
        ax[0].patch.set(lw=6, ec='b')
        ax[0].set_axis_on()            
        plt.show()