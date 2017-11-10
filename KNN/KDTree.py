
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

##### KNN by KDTree ######
'''
See [1] for more details
'''
class KNode():
    def __init__(self,data=None,left=None,right=None,axis=None,median=0):
        '''Args:
        data:data located on the split plane data[-1] is label
        left:left child
        Right:right child
        axis:current axis to split data
        median:value to split data set
        '''
        self.data=data[:-1]
        self.label=data[-1]
        self.left=left
        self.right=right
        self.axis=axis
        #self.sel_axis=sel.axis
        self.median=median
        
    def search_node(self,point,k,results,get_dist):
        if not self:
            return
        nodeDist=get_dist(self)
        results.enqueue((self,nodeDist))
        if self.axis=='Leaf':
            return
        split_plane=self.data[self.axis]
        plane_dist=point[self.axis]-split_plane
        plane_dist2=plane_dist**2
        
        if point[self.axis]<split_plane:
            if self.left is not None:
                self.left.search_node(point,k,results,get_dist)
        else:
            if self.right is not None:
                self.right.search_node(point,k,results,get_dist)
        
        if plane_dist2<results.max_() or results.size()<k:
            if point[self.axis]<self.data[self.axis]:
                if self.right is not None:
                    self.right.search_node(point,k,results,get_dist)
            else:
                if self.left is not None:
                    self.left.search_node(point,k,results,get_dist)                    

##Create KD tree
def create(point_list,axis=0,sel_axis=None):
    if sel_axis is None:
        raise ValueError('you must define a function for sel_axis. It decides how you choose the axis at each time of tree split')
    m=len(point_list)
    if m==0:
        return None
    if m==1:
        return KNode(data=point_list.reshape(point_list.shape[1],),axis='Leaf',median=point_list[:,axis])
    curr_=np.array(sorted(point_list,key=lambda p:p[axis]))
    median=curr_[m//2,axis]
    left=curr_[:m//2]
    right=curr_[m//2+1:]
    loc=curr_[m//2]
    node=KNode(data=loc,axis=axis,median=median)
    node.left=create(left,axis=sel_axis(axis),sel_axis=sel_axis)
    node.right=create(right,axis=sel_axis(axis),sel_axis=sel_axis)
    return node

### Use Heap to store the top K neighbours
class PQ:
    def __init__(self,k,heap=[]):
        self.k=k
        self.heap=list(heap)
    
    def enqueue(self,e):
        size=self.size()
        if size==self.k:
            if e[1]<self.max_():
                self.heap[0]=e
                self.sftdown(e,0)
        else:
            self.heap.append(e)
            if size>0:
                self.sftup(e,size)
    
    def sftup(self,e,size):
        i,j=size,self.parent(size)
        curr_d=e[1]
        p_d=self.dist(j)
        while j>=0 and curr_d>self.dist(j):
            self.heap[i]=self.heap[j]
            i,j=j,self.parent(j)
        self.heap[i]=e
            
    def sftdown(self,e,idx):
        curr_d=e[1]    
        i,j=idx,self.left(idx)
        while j+1<=self.k-1:
            if self.dist(j+1)>self.dist(j):
                j=j+1
            if curr_d<self.dist(j):
                self.heap[i]=self.heap[j]
                i,j=j,self.left(j)
            else:
                break
        self.heap[i]=e
        
    def size(self):
        return len(self.heap)
    def parent(self,index):
        return (index-1)//2  
    def items(self):
        return self.heap
    def left(self,index):
        return index*2+1
    def right(self,index):
        return index*2+2
    def dist(self,index):
        return self.heap[index][1]
    def max_(self):
        return self.heap[0][1]


'''
Args:
   tree:KD tree
   point: test sample
   k:how many neighbours used
   dist:function to caculate distance. Default is square distance
Returns:
   neightbous
'''
def search_knn(tree,point,k,dist=None):
    if dist is None:
        get_dist=lambda obj:np.mean((point-obj.data)**2)
    else:
        get_dist=lambda obj:dist(obj.data,point)
        
    results=PQ(k)
    tree.search_node(point,k,results,get_dist)
    
    BY_VALUE=lambda kv:kv[1]
    return sorted(results.items(),key=BY_VALUE)

#predict one sample
#return predicted label
def predict_one(tree,test,num):
    res=search_knn(tree,test,k=num)
    k=[i[0].label for i in res]
    y_hat=np.argmax(np.bincount(k))
    return y_hat 

#predict a gourp of samples
#Return predictedlabel
def predict(tree,test,num=20):
    res=[]
    for i in test:
        res.append(predict_one(tree,i,num))
    return res

#display neighbours
def visual(tree,test):
    res=search_knn(tree,test,20,dist=None)
    plt.figure(figsize=(10,6))
    plt.subplot(2,1,1)
    plt.imshow(test.reshape(28,28),cmap='gray')
    plt.title('test sample')
    plt.subplot(2,1,2)
    canvas=np.zeros((4*28,5*28))
    for i in range(len(res)):
        canvas[i//5*28:i//5*28+28,i%5*28:i%5*28+28]=res[i][0].data.reshape(28,28)
    plt.imshow(canvas,cmap='gray')
    plt.title('{} Neighbour'.format(20))
    plt.show()

train=pd.read_csv('C:/Users/jiema/Documents/data/Mnist/train.csv')
test=pd.read_csv('C:/Users/jiema/Documents/data/Mnist/test.csv')

train_data=train.iloc[:,1:].values
train_label=train.iloc[:,:1].values
test_data=test.values
train=np.hstack((train_data,train_label))

if __name__=='__main__':
    import time
    it=time.time()
    tree=create(train,axis=0,sel_axis=lambda idx:(idx+1)%784)
    it=time.time()-it
    print('### Training Stage ###\nTraining {} samples require {} seconds\n'.format(it,len(train)))
    it=time.time()
    result=predict(tree,test_data[:1],num=20)
    it=time.time()-it
    print('### Inference Stage ###\nInference time: {} seconds/sample'.format(it))

    sample=test_data[np.random.randint(len(test_data))]
    visual(tree,sample)

