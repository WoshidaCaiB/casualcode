'''Simple implementation of Kmeans'''
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

class KMeans:
    '''
	Arg:
	   distance: the function defines how to caculate the distance Default: E-distance
	'''
    def __init__(self,distance=None):
        self.distance=distance
        if distance is None:
            self.distance=lambda x,y:np.mean((x-y)**2)     
            
    def cluster(self,center,m):
        cluster=[[] for _ in range(m)]
        for data in self.data:
            mindistance=0.
            for i in range(m):
                curr_distance=self.distance(data,center[i])
                if mindistance==0.:
                    mindistance=curr_distance
                    index=i
                if curr_distance<mindistance:
                    mindistance=curr_distance
                    index=i
            cluster[index].append(data)
        newcenter=[]
        for i in range(m):      
            newcenter.append(np.mean(cluster[i],axis=0))
        return newcenter 
    
    def run(self,x,center_num):
        self.data=x
        m=len(self.data)
        center_idx=[]
        center=[]
        for j in range(center_num):
            idx=np.random.randint(0,m)
            while idx in center_idx:
                idx=np.random.randint(0,m)
            center_idx.append(idx)
            center.append(self.data[idx])
        while True:
            new_center=self.cluster(center,center_num)
            update=np.mean([np.linalg.norm(center[i]-new_center[i],2) for i in range(center_num)])
			### Stop condition: if the E-distance between updated clusters and old clusters is small enough, update stops
            if update<0.0000001:
                break
            center=new_center
        return new_center
    
    def label(self,center):
        label_vec=[]
        for data in self.data:
            mindistance=0.
            index=0
            for i in range(len(center)):
                curr_dis=self.distance(data,center[i])
                if mindistance==0.:
                    mindistance=curr_dis
                if mindistance>curr_dis:
                    mindistance=curr_dis
                    index=i
            label_vec.append(index)
        return label_vec

    def plot(self,center,label):
        center=np.array(center)
        label=np.array(label)
        plt.scatter(self.data[:,0],self.data[:,1],c=label)
        plt.scatter(center[:,0],center[:,1],s=100,marker='o',alpha=1.,c='k')
        plt.show()


def test(obj,m):
    plt.figure(figsize=(12,9))
    for i in range(m):
        x,y=make_blobs(n_samples=1000,n_features=2,centers=3,cluster_std=0.5,shuffle=True)
        plt.subplot(m,2,2*i+1)
        plt.title('Sample')
        plt.scatter(x[:,0],x[:,1],c=y)
        plt.subplot(m,2,2*i+2)
        plt.title('Result')
        center=obj.run(x,3)
        label=obj.label(center)
        center=np.array(center)
        label=np.array(label)
        plt.scatter(x[:,0],x[:,1],c=label)
        plt.scatter(center[:,0],center[:,1],s=100,marker='o',alpha=1.,c='k')
    plt.show()

if __name__=='__main__':
    model=KMeans()
    test(model,3)
