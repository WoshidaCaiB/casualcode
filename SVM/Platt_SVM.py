
# coding: utf-8

import numpy as np
class ACSVM:
    '''Platt SVM 
    SMO is used for training to update Alpha values. 
    Update procedure contains 2 loops. 
    Out loop runs through the entire alpha set to update alpha values 
    Inner loop only pays attention to those alpha values (0<a<C) 
    The alpha values which does not meet KKT condition will be updated
    KKT condition:
       1. 0<alpha<C and distance=0
       2. alpha=0 and distance>1-epsilon
       3. alpha=C and distance<1-epsilon
    Args:
       model: chose kernel function "linear", "Poly" or "rbf"
       C: compensate (0<=a<=C) less C means less penalty on wrong classifcations
       degree: play different roles in different kernel function
       epsilon: trade-off on the distance 
    '''
    def __init__(self,model,degree=1.,c=1.,epsilon=0.00001):
        if model=='linear':
            self._kernel=self._kernel_linear
        elif model=='rbf':
            self._kernel=self._kernel_rbf
        elif model=='poly':
            self._kernel=self._kernel_poly
        else:
            raise ValueError('wrong kernel type. Please select kernel type from "poly", "rbf" or "linear"')
        self._mode=model
        self.C=c #compensate
        self.degree=degree #used for kernel function
        self.epsilon=epsilon #precision
    
    def _prepare(self,x,y):
        '''
        Initialize alpha and bias
        Calculate Gram matrix - kernel_function(x,y)
        '''
        self._data=np.array(x)
        self._N=self._data.shape[0]
        self._label=np.array(y)
        self._gram=self._kernel(x,x,self.degree)
        self.alpha=np.zeros(self._data.shape[0])
        self._E_cache=np.zeros((self._N,2))
        self.b=0.
        
    def _CalcE(self,i):
        '''calculate error: (wx+b)-y'''
        return np.dot(self._label*self.alpha,self._gram[:,i])+self.b-self._label[i]
    
    def _a2_index(self,i,Ei):
        '''
        chose the second alpha
        The Alpha which gives the highest |E1-E2| will be picked
        '''
        self._E_cache[i]=[1,Ei]
        validE=np.nonzero(self._E_cache[:,0])[0]
        if(len(validE)>1):
            j=0
            maxDelta=0
            Ej=0
            for k in validE:
                if(k==i):
                    continue
                Ek=self._CalcE(k)
                if(abs(Ei-Ek)>maxDelta):
                    j=k
                    maxDelta=abs(Ei-Ek)
                    Ej=Ek
        
        else:
            j=self._selectJ(i,self._N)
            Ej=self._CalcE(j)
        return j,Ej
    
    def _selectJ(self,i,N):
        j=i
        while(j==i):
            j=np.random.choice(N)
        return j
    
    def _updateE(self,k):
        Ek=self._CalcE(k)
        self._E_cache[k]=[1,Ek]
        
    def _get_bound(self,i,j,a,b):
        if self._label[i]!=self._label[j]:
            L=max(0,b-a)
            H=min(self.C,self.C+b-a)
        else:
            L=max(0,a+b-self.C)
            H=min(self.C,a+b)
        return (L,H)
    
    def _inner(self,i):
        #inner loop
        E1=self._CalcE(i)
        if ((self._label[i]*E1>self.epsilon) and (self.alpha[i]>0.)) or ((self._label[i]*E1<-self.epsilon) and (self.alpha[i]<self.C)):
            j,E2=self._a2_index(i,E1)
            a1old=float(self.alpha[i])
            a2old=float(self.alpha[j])
            eta=self._gram[i][i]+self._gram[j][j]-2.*self._gram[i][j]
            if eta<=0:
                return 0
            a2new=a2old+self._label[j]*(E1-E2)/eta
            L,H=self._get_bound(i,j,a1old,a2old)
            a2new=min(H,max(L,a2new))
            self._updateE(j)
            if abs(a2new-a2old)<0.00001:
                return 0
            a1new=a1old+self._label[i]*self._label[j]*(a2old-a2new)
            self._updateE(i)
            b1=self.b-E1-self._label[i]*self._gram[i][i]*(a1new-a1old)-self._label[j]*self._gram[j][i]*(a2new-a2old)
            b2=self.b-E2-self._label[i]*self._gram[i][j]*(a1new-a1old)-self._label[j]*self._gram[j][j]*(a2new-a2old)
            if a1new<self.C and a1new>0:
                self.b=b1
            elif a2new<self.C and a2new>0:
                self.b=b2
            else:
                self.b=(b1+b2)/2
            self.alpha[i]=a1new
            self.alpha[j]=a2new
            return 1
        else:
            return 0

    def _findSupport(self):
        res=[]
        for i in range(len(self.alpha)):
            if self.alpha[i]>0 and self.alpha[i]<self.C:
                res.append(i)
        return res
    
    def fit(self,x,y,maxiter):      
        self._prepare(x,y)
        entire=True
        iters=0
        #the algorithm runs between inner loop and out loop to update alpha
        while(iters<maxiter):
            if entire:
                AlphaChange=0
                for i in range(self._N):
                    AlphaChange+=self._inner(i)
                if AlphaChange==0:
                    iters+=1
                    continue
                else:
                    iters=0
                    entire=False
            else:
                AlphaChange=0
                bound=self._findSupport()
                for i in bound:
                    AlphaChange+=self._inner(i)
                if AlphaChange==0:
                    iters+=1
                    entire=True 
                else:
                    iters=0
    
    def visualize(self,x,y,sigma):
        positive=x[y==1]
        negative=x[y==-1]
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.scatter(positive[:,0],positive[:,1],c='r',marker='o')
        plt.scatter(negative[:,0],negative[:,1],c='g',marker='o')
        nonZeroAlpha=self.alpha[np.nonzero(self.alpha)]
        supportVector=self._data[np.nonzero(self.alpha)]
        y=self._label[np.nonzero(self.alpha)]
        plt.scatter(supportVector[:,0],supportVector[:,1],s=100,c='y',alpha=1.,marker='o')
        print('支持向量个数:',len(nonZeroAlpha))
        max1=max(x[:,0])
        min1=min(x[:,0])
        max2=max(x[:,1])
        min2=min(x[:,1]) 
        x1=np.arange(0.8*min1,1.2*max1,0.1)
        x2=np.arange(0.8*min2,1.2*max2,0.1)
        X1,X2=np.meshgrid(x1,x2)
        g=self.b
        for i in range(len(nonZeroAlpha)):
            if self._mode=='rbf':
                g+=nonZeroAlpha[i]*y[i]*np.exp(-0.5*((X1-supportVector[i][0])**2+(X2-supportVector[i][1])**2)/(self.degree**2))
            elif self._mode=='linear':
                g+=nonZeroAlpha[i]*y[i]*(X1*supportVector[i][0]+X2*supportVector[i][1])
                w=self.w
                y1=-(x1*w[0]+self.b)/w[1]
                plt.plot(x1,y1)
                
        plt.contour(X1,X2,g,0,colors='b',label='classification bound')#the classification curve
        plt.title("sigma: %f" %self.degree)
        plt.show()
    
    def predict(self,y):
        return np.sign(np.dot(self._label*self.alpha,self._kernel(self._data,y,self.degree))+self.b)
    
    def _kernel_poly(self,x,y,p):
        return (x.dot(y.T)+1)**p
    
    def _kernel_rbf(self,x,y,gamma):
        return np.exp(-0.5*np.sum((x[...,None,:]-y)**2,axis=2)/(gamma)**2)
    
    def _kernel_linear(self,x,y,scale):
        return x.dot(y.T)*scale
    
    @property
    def w(self):
        return np.dot((self._label*self.alpha),self._data)


import numpy as np
from sklearn.datasets import make_blobs, make_circles
import matplotlib.pyplot as plt
x, y = make_blobs(n_samples=400,n_features=2,centers=2,cluster_std=1,shuffle=True)
y[y==0]=-1
x.astype(np.float32)
y.astype(np.float32)
plt.scatter(x[y==1,0],x[y==1,1],c='r')
plt.scatter(x[y==-1,0],x[y==-1,1],c='b')
plt.show()

#linear SVM
a=ACSVM(model='linear',c=1,degree=1.)
a.fit(x,y,40)
y_pred=a.predict(x)
print(np.sum(y_pred==y))
a.visualize(x,y,2)


x, y = make_circles(n_samples=400, factor=.3, noise=.05)
y[y==0]=-1
x.astype(np.float32)
y.astype(np.float32)
plt.scatter(x[y==1,0],x[y==1,1],c='r')
plt.scatter(x[y==-1,0],x[y==-1,1],c='b')
plt.show()

#rbf kernel
a=ACSVM(model='rbf',c=1,degree=1.)
a.fit(x,y,40)
y_pred=a.predict(x)
print(np.sum(y_pred==y))
a.visualize(x,y,2)

