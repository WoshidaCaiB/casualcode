# Casualcode-ML

casual code for machine learning algorithms...

# 1. Platt SVM

Implementation of Support Vector Machine (SVM). SMO is used to as training algorithm

To run the model Python 3.6 is required

The kernel included is linear, poly and rbf

Linear kernel Result:

![img](https://github.com/WoshidaCaiB/casualcode/blob/master/SVM/output_2_1.png)

RBF kernel Result:

![img](https://github.com/WoshidaCaiB/casualcode/blob/master/SVM/output_4_1.png)

The yellow points are support vectors

# 2. CART 

Implementation of CART tree. Gini is the criterion for node splitting

Model can deal with both catergorical and continous values

The code for tree visualization is from [1]

#Use the example data from <<统计学习原理>> to test model

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



    #### CART Tree structure ####
	
     {'House': {(0, '是'): 'yes', (1, 'not 是'): {'Work': {(0, '是'): 'yes', (1, 'not 是'): 'no'}}}}
	 
    #### Predicted Label ####
	
     ['no', 'no', 'yes', 'yes', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']
	 
    #### Leaf Num: 3 ####
	
    #### Tree Depth: 2 ####
    
    Tree Structure:
    
   ![img](https://github.com/WoshidaCaiB/casualcode/blob/master/CART/output_3_1.png)

#Use the example data from <<机器学习>> for test. This dataset contains continuous values. Because tree visualization code does not support chinese charater display. I have to use Pinyin to represent the feature value

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

    #### CART Tree structure ####
	
     {'wen li': {(0, 'qing xi'): {'mi du': {(0, '<0.3815'): 'no', (1, '>0.3815'): 'yes'}}, (1, 'not qing xi'): {'se ze': {(0, 'wu hei'): 
     
     {'gen di': {(0, 'quan suo'): 'yes', (1, 'not quan suo'): 'no'}}, (1, 'not wu hei'): 'no'}}}}
	 
    #### Predict Labels ####
	
     ['yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'no']
	 
    #### Leaf Num: 5 ####
	
    #### Tree Depth: 3 ####
	
    Tree Structure:
    
   ![img](https://github.com/WoshidaCaiB/casualcode/blob/master/CART/output_4_1.png)
    
Reference:

[1]《机器学习实战》

# 3. KMeans

simple implementation of kmeans. Results are highly sensitive to the initial values of the cluster points

Results:

![img](https://github.com/WoshidaCaiB/casualcode/blob/master/Kmeans/Figure_1.png)

# 4. KNN

Implementation of KNN based on KDTree. See [1] for more details on KDTree

Taking the advantage of KDTree, both training and prediction go much faster

(### Training Stage ###)

Train time: 3.7246341705322266 seconds (42000 samples included)

(### Inference Stage ###)

Inference time: 1.5805480480194092 seconds/sample

Test model on Mnist classification

Results:

![img](https://github.com/WoshidaCaiB/casualcode/blob/master/KNN/res1.png)
![img](https://github.com/WoshidaCaiB/casualcode/blob/master/KNN/res2.png)

![img](https://github.com/WoshidaCaiB/casualcode/blob/master/KNN/res3.png)
![img](https://github.com/WoshidaCaiB/casualcode/blob/master/KNN/res4.png)

Reference:

[1]https://github.com/stefankoegl/kdtree
