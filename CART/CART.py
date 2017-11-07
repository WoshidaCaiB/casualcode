
# coding: utf-8

class CART:
    '''***Cart Tree for classification task***
    This is to build Cart tree for classification work. 
    Model is able to deal with both categorical and continuous feature values
    Gini is used as the criteron for node split. 
    Cart tree is always binary tree
    TO DO:
       add prune tech
    '''
    
    def _continuouslabel(self,i):
        '''
        ceate feature set for continous feature values for node split
        sort the value first and then use the average of the 2 consecutive values as the feature set for node split
        '''
        feat=set(i)
        newfeat=list(feat)
        m=len(newfeat)-1
        newfeat.sort()
        label=[(newfeat[j]+newfeat[j+1])/2. for j in range(m)]
        return label
    
    def _CalcProb(self,labelV):
        dim=len(labelV)
        labelCount={}
        prob=0
        for i in labelV:
            if i not in labelCount.keys():
                labelCount[i]=0
            labelCount[i]+=1
        for key in labelCount.keys():
            prob+=(labelCount[key]/dim)**2
        return 1-prob

    def _CalcGini(self,data,i,feat,discrete):
        #calculate Gini value for the proposed feature: i, split node: feat
        if not discrete:
            class1=[example[-1] for example in data if example[i]<feat]
            class2=[example[-1] for example in data if example[i]>=feat]  
        else:
            class1=[example[-1] for example in data if example[i]==feat]
            class2=[example[-1] for example in data if example[i]!=feat] 
        dim1=len(data)
        dim2=len(class1)
        prob=dim2/dim1
        Gini=prob*self._CalcProb(class1)+(1-prob)*self._CalcProb(class2)
        return Gini
    
    def _BestFeaturetoSplit(self,data):
        #find the best split node according to Gini values
        minGini=1.
        n=len(data[0])
        for i in range(n-1):
            if isinstance(data[0][i],float):
                is_discrete=False
                featurevector=[example[i] for example in data]
                feature=self._continuouslabel(featurevector)
            else:
                is_discrete=True
                featurevector=[example[i] for example in data]
                feature=set(featurevector)
            for feat in feature:
                Gini=self._CalcGini(data,i,feat,is_discrete)
                if Gini<minGini:
                    minGini=Gini
                    splitindex=i
                    splitfea=feat
        Bestfeat=splitindex
        is_discrete=not isinstance(data[0][Bestfeat],float)
        return Bestfeat,splitfea,is_discrete
    
    def _splitData(self,data,i,feat,discrete):
        #Given the feature i, split node: feat, split the data into data1 and data2 
        data1=[]
        data2=[]
        
        if not discrete: 
            for example in data:
                if example[i]<feat:
                    data1.append(example)
                   
                else:
                    data2.append(example)
            
            featurevector1=[frame[i] for frame in data1]
            featurevector2=[frame[i] for frame in data2] 
            label1=self._continuouslabel(featurevector1)
            label2=self._continuouslabel(featurevector2)
            if label1==[]:
                data1=self._remove_i(data1,i)
            if label2==[]:
                data2=self._remove_i(data2,i)
                             
        else:
            for example in data:
                if example[i]==feat:
                    reducedvector=example[0:i]
                    reducedvector.extend(example[i+1:])
                    data1.append(reducedvector)
                else:
                    data2.append(example)
            featurevector2=[frame[i] for frame in data2] 
            if len(set(featurevector2))==1:
                data2=self._remove_i(data2,i)

        return data1,data2
        
    def _remove_i(self,data,i):
        resData=[]
        for k in data:
            curr=k[0:i]
            curr.extend(k[i+1:])
            resData.append(curr)
        return resData
    
    def _major(self,classList):
        classCount={}
        for vote in classList:
            if vote not in classCount.keys():
                classCount[vote]=0
            classCount[vote]+=1
        sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
        return sortedClassCount[0][0]  
        
    def createtree(self,x,labels):
        '''
        Set-up the CART tree recursively 
        Args:
           x: training data
           labels: a list contain feature names
        ''' 
        labelvector=[example[-1] for example in x]
        m=len(x[0])
        if len(x)==1:
            return x[0][-1]
        if labelvector.count(labelvector[0])==len(labelvector):
            return labelvector[0]
        if m==1:
            return self._major(labelvector)
        bestfeature_index,bestfeatval,is_discrete=self._BestFeaturetoSplit(x)
        feature=labels[bestfeature_index]
        mytree={feature:{}}
        subdata1,subdata2=self._splitData(x,bestfeature_index,bestfeatval,is_discrete)
        
        label1=labels.copy()
        label2=label1.copy()
        
        if len(subdata1[0])<m:
            del(label1[bestfeature_index])
        if len(subdata2[0])<m:
            del(label2[bestfeature_index])
            
        if not is_discrete:
            key1=(0,'<'+str(bestfeatval))
            key2=(1,'>'+str(bestfeatval))
        else:
            key1=(0,str(bestfeatval))
            key2=(1,'not '+str(bestfeatval))
    
        mytree[feature][key1]=self.createtree(subdata1,label1)
        mytree[feature][key2]=self.createtree(subdata2,label2)
        return mytree
    
    def predict(self,mytree,data,label):
        result=[]
        for sample in data:
            classres=self._predictsample(mytree,sample,label)
            result.append(classres)
        return result

    def _predictsample(self,mytree,sample,label):
        '''
        use the tree to classify the sample
        Args:
           mytree: the tree just built
           sample: test data
           label:a list contain the feature names
        Return:
           the predicted class for sample
        '''        
        firstval=list(mytree.keys())[0]
        secondDict=mytree[firstval]
        index=label.index(firstval)
        value=sample[index]
        keyVec=list(secondDict.keys())
        leftindex,rightindex=keyVec[0],keyVec[1]
        if isinstance(value,float):
            if eval(str(value)+leftindex[1]):
                nextdict=secondDict[leftindex]
            else:
                nextdict=secondDict[rightindex]
        else:
            if keyVec[0][0]==1:
                leftindex,rightindex=keyVec[1],keyVec[0]
            if (str(value))==keyVec[0][1]:
                nextdict=secondDict[leftindex]
            else:
                nextdict=secondDict[rightindex]
        if isinstance(nextdict,dict):
            classval=self._predictsample(nextdict,sample,label)
        else:
            classval=nextdict
        return classval

def getNumLeafs(mytree):
    #Given the tree, return the number of leafs
    num=0
    first=list(mytree.keys())[0]
    second=mytree[first]
    for i in list(second.keys()):
        value=mytree[first][i]
        if isinstance(value,dict):
            num+=getNumLeafs(value)
        else:
            num=num+1
    return num

def getTreeDepth(mytree):
    #Given the tree, return the depth of the tree (root is depth 0)
    maxDepth=0
    firstStr=list(mytree.keys())[0]
    second=mytree[firstStr]
    for key in second.keys():
        if isinstance(second[key],dict):
            thisDepth=getTreeDepth(second[key])+1
        else:
            thisDepth=1
        if thisDepth>maxDepth:
            maxDepth=thisDepth
    return maxDepth

#plot the tree. Code is copied from <<Python与机器学习实践>>
import matplotlib.pyplot as plt

decisionNode=dict(boxstyle='sawtooth',fc='0.8')
leafNode=dict(boxstyle='round4',fc='0.8')
arrow_args=dict(arrowstyle='<-')

def plotNode(nodeTxt,centerPt,parentPt,nodeType):
    createPlot.ax1.annotate(nodeTxt,xy=parentPt,xycoords='axes fraction',
                           xytext=centerPt,textcoords='axes fraction',
                          va='center',ha='center',bbox=nodeType,arrowprops=arrow_args)
def plotMidText(cntrPt,parentPt,txtString):
    xMid=(parentPt[0]-cntrPt[0])/2.+cntrPt[0]
    yMid=(parentPt[1]-cntrPt[1])/2.+cntrPt[1]
    createPlot.ax1.text(xMid,yMid,txtString)
def plotTree(myTree,parentPt,nodeTxt):
    numLeafs=getNumLeafs(myTree)
    depth=getTreeDepth(myTree)
    firstStr=list(myTree.keys())[0]
    cntrPt=(plotTree.xOff+(1.0+float(numLeafs))/2./plotTree.totalW,plotTree.yOff)
    plotMidText(cntrPt,parentPt,nodeTxt)
    plotNode(firstStr,cntrPt,parentPt,decisionNode)
    secondDict=myTree[firstStr]
    plotTree.yOff=plotTree.yOff-1./plotTree.totalD
    for key in secondDict.keys():
        if isinstance(secondDict[key],dict):
            plotTree(secondDict[key],cntrPt,str(key[1]))
        else:
            plotTree.xOff=plotTree.xOff+1./plotTree.totalW
            plotNode(secondDict[key],(plotTree.xOff,plotTree.yOff),cntrPt,leafNode)
            plotMidText((plotTree.xOff,plotTree.yOff),cntrPt,str(key[1]))
    plotTree.yOff=plotTree.yOff+1./plotTree.totalD
    
def createPlot(inTree):
    fig=plt.figure(1,facecolor='white')
    fig.clf()
    axprops=dict(xticks=[],yticks=[])
    createPlot.ax1=plt.subplot(111,frameon=False,**axprops)
    plotTree.totalW=float(getNumLeafs(inTree))
    plotTree.totalD=float(getTreeDepth(inTree))
    plotTree.xOff=-0.5/plotTree.totalW
    plotTree.yOff=1.
    plotTree(inTree,(0.5,1.),' ')
    plt.show()

#Use the example data from <<统计学习原理>> to test our model

x=[['青年','否','否','一般','no'],
   ['青年','否','否','好','no'],
  ['青年','是','否','好','yes'],
  ['青年','是','是','一般','yes'],
  ['青年','否','否','一般','no'],
  ['中年','否','否','一般','no'],
  ['中年','否','否','好','no'],
  ['中年','是','是','好','yes'],
  ['中年','否','是','非常好','yes'],
  ['中年','否','是','非常好','yes'],
  ['老年','否','是','非常好','yes'],
  ['老年','否','是','好','yes'],
  ['老年','是','否','好','yes'],
  ['老年','是','否','非常好','yes'],
  ['老年','否','否','一般','no']]

a=CART()
label=['Age','Work','House','Credit']
mytree1=a.createtree(x,label)
print('#### CART Tree structure ####\n',mytree1)
result=a.predict(mytree1,x,label)
print('#### Predicted Label ####\n',result)
print('#### Leaf Num: {} ####'.format(getNumLeafs(mytree1)))
print('#### Tree Depth: {} ####\n'.format(getTreeDepth(mytree1)))   
createPlot(mytree1)

#use the example data from <<机器学习>> for test. This dataset contains continuous values

watermelon=[['qing lv','quan suo','zhuo xiang','qing xi','ao xian','ying hua',0.697,0.460,'yes'],
           ['wu hei','quan suo','chen men','qing xi','ao xian','ying hua',0.774,0.376,'yes'],
           ['wu hei','quan suo','zhuo xiang','qing xi','ao xian','ying hua',0.634,0.264,'yes'],
           ['qing lv','quan suo','chen men','qing xi','ao xian','ying hua',0.608,0.318,'yes'],
           ['qian bai','quan suo','zhuo xiang','qing xi','ao xian','ying hua',0.556,0.215,'yes'],
           ['qing lv','quan suo','zhuo xiang','qing xi','shao ao','ruan nian',0.403,0.237,'yes'],
           ['wu hei','quan suo','zhuo xiang','shao hu','shao ao','ruan nian',0.481,0.149,'yes'],
           ['wu hei','shao quan','zhuo xiang','qing xi','shao ao','ying hua',0.437,0.211,'yes'],
           ['wu hei','shao quan','chen men','mo hu','shao ao','ying hua',0.666,0.091,'no'],
           ['qing lv','ying ting','qing cui','qing xi','ping tan','ruan nian',0.243,0.267,'no'],
           ['qian bai','ying ting','qing cui','mo hu','ping tan','ying hua',0.245,0.057,'no'],
           ['qian bai','quan suo','zhuo xiang','mo hu','ping tan','ruan nian',0.343,0.099,'no'],
           ['qing lv','shao quan','zhuo xiang','shao hu','ao xian','ying hua',0.639,0.161,'no'],
           ['qian bai','shao quan','chen men','mo hu','ao xian','ying hua',0.657,0.198,'no'],
           ['wu hei','shao quan','zhuao xiang','qing xi','shao ao','ruan nian',0.360,0.370,'no'],
            ['qian bai','quan suo','zhuo xiang','mo hu','ping tan','ying hua',0.593,0.042,'no'],
            ['qing lv','quan suo','chen men','shao hu','shao ao','ying hua',0.719,0.103,'no']]

b=CART()
labels=['se ze','gen di','qiao sheng','wen li','qi bu','chu gan','mi du','han tang lv']
mytree2=b.createtree(watermelon,labels)
print('#### CART Tree structure ####\n',mytree2)
result=a.predict(mytree2,watermelon,labels)
print('#### Predict Labels ####\n',result)
print('#### Leaf Num: {} ####'.format(getNumLeafs(mytree2)))
print('#### Tree Depth: {} ####\n'.format(getTreeDepth(mytree2)))   
createPlot(mytree2)
