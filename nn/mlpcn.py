# -*- coding: utf-8 -*-
"""
Description: The class mlpcn implements a one-dimensional perceptron with two
neuron layers. The user can specify the output function (linear, logistic, 
softmax), the training type (sequential, batch), the number of hidden neurons 
and the learning rate.
"""

import numpy as np  #numerical package for scientific computing

class mlpcn: 
    """ class for a multi-layer perceptron """    
    # Notation for the algorithm of a two layer perceptron
    # inputs: xi 
    # first weights: w1ij, 
    # activation functions: g1(), g2() = linear, logist, softmax
    # first outputs: o1j = g1(xi*w1ij)
    # second weights: w2jk,
    # second outputs: o2k = g2(o1j*w2jk)
    # targets: tk
    # error function: E = 1/2 (tk-o2k)^2
    # first layer: dE/dw1ij = (o2k-tk)*g2'(o1i*w2ik)*g1'(xi*w1ij)*w2jk
    # second layer: dE/dw2jk = (o2k-tk)*g2'(o1i*w2ik)*o1j 
    
    def __init__(self,inputs,targets,nhn,eta,functype,traintype):
        """ constructor """
        # inputs = N1xN2 matrix -> add bias node later -> N1x(N2+1) matrix
        # targets = N1xN3 matrix
        # nhn = number of hidden neurons (excluding bias node)
        # weights1 = (N2+1)x(nhn) matrix
        # weights2 = (nhn+1)x(N3) matrix
        
        #add -1 bias node for inputs
        self.x = np.concatenate((-np.ones((len(inputs),1)),inputs),axis=1) 
        
        #targets
        self.t = targets  
        
        #sequential or batch training
        self.traintype = traintype
        
        #initialize weights of first and second layer
        #random numbers in range [-1/sqrt(n),1/sqrt(n)] with n=number of weights to a neuron
        self.w1 = (np.random.rand(np.shape(self.x)[1],nhn)*2-1)/np.sqrt(np.shape(self.x)[1])
        self.w2 = (np.random.rand(nhn+1,np.shape(self.t)[1])*2-1)/np.sqrt(nhn+1) 
        #self.w1 = np.array([[0.51652025,-0.19428886,0.68459804],[0.55135733,-0.61364545,0.09567272]]) 
        #self.w2 = np.array([[-0.20161876],[ 0.44428779],[-0.20328278],[ 0.20249294]]) 
        
        #learning rate
        self.eta = eta
        
        #set activation function type for output neurons 
        self.functype = functype
        
        print('Two-layer Perceptron')
        print('with',nhn,'hidden and',np.shape(self.t)[1],'output neurons')
        print('with',np.shape(self.x)[0],'inputs / targets of dim',np.shape(self.x)[1]-1,'/',np.shape(self.t)[1])
        print('with',self.functype,'function for output neurons')
        print('with learning rate',self.eta,'and',self.traintype,'training')
        
        #print('weight1:\n',self.w1)
        #print('weight2:\n',self.w2)
        
    def pcnfwd(self,inputs,weights,functype):
        """ output of one neuron layer """
        if functype == 'linear':
            #linear function
            func = lambda x: x 
        elif functype == 'sigmoid':
            #sigmoid
            func = lambda x: 1/(1+np.exp(-x)) 
        elif functype == 'softmax':
            #softmax
            func = lambda x: np.exp(x)/np.sum(np.exp(x),axis=-1,keepdims=True)
            
        return func(np.dot(inputs,weights))    
    
    def mlpfwd(self,inputs,addbias):
        """ calculate the outputs of each layer """
        #add biad node (optional)
        if addbias == True:
            inputs = np.concatenate((-np.ones((len(inputs),1)),inputs),axis=1) 
        
        #output of first layer using the sigmoid function
        if inputs.ndim == 1:
            o1 = np.concatenate(([-1],self.pcnfwd(inputs,self.w1,'sigmoid')))  
        elif inputs.ndim == 2:
            o1 = np.concatenate((-np.ones((len(inputs),1)),self.pcnfwd(inputs,self.w1,'sigmoid')),axis=1)  
        
        #output of second layer
        o2 = self.pcnfwd(o1,self.w2,self.functype)
        return o1,o2       
    
    def errfunc(self,outputs,targets):
        """ error function """
        return 1/2*np.trace(np.dot(np.transpose(targets-outputs),targets-outputs))
    
    def dw(self,x,o1,o2,t,w1,w2):
        """ calculate weights for hidden and output layers """
        
        #linear output function
        if self.functype == 'linear':
            #dw2jk = o1j*(tk-o2k)
            #dw1ij = xi*dw2jk*w2jk*(1-o1j)
            if x.ndim == 1:
                dw2 = np.outer(np.transpose(o1),t-o2)
                dw1 = np.outer(np.transpose(x),np.dot(t-o2,np.transpose(w2))*o1*(1.0-o1))
            elif x.ndim == 2:
                dw2 = np.dot(np.transpose(o1),t-o2)
                dw1 = np.dot(np.transpose(x),np.dot(t-o2,np.transpose(w2))*o1*(1.0-o1))
        
        #sigmoid,softmax output function
        elif self.functype == 'sigmoid' or self.functype == 'softmax':
            #dw2jk = o1j*(tk-o2k)*o2k*(1-o2k)
            #dw1ij = xi*dw2jk*w2jk*(1-o1j)
            if x.ndim == 1:
                dw2 = np.outer(np.transpose(o1),(t-o2)*o2*(1.0-o2))
                dw1 = np.outer(np.transpose(x),np.dot((t-o2)*o2*(1.0-o2),np.transpose(w2))*o1*(1.0-o1))
            elif x.ndim == 2:
                dw2 = np.dot(np.transpose(o1),(t-o2)*o2*(1.0-o2))
                dw1 = np.dot(np.transpose(x),np.dot((t-o2)*o2*(1.0-o2),np.transpose(w2))*o1*(1.0-o1))
                
        #delete column corresponding to bias node
        dw1 = np.delete(dw1,0,1)
        
        #divide by the number of input vectors to get the average (batch case)
        if x.ndim == 2:
            dw1 = dw1 / len(x)
            dw2 = dw2 / len(o1)
               
        #multiply with learning rate eta
        dw1 *= self.eta
        dw2 *= self.eta
        
        return dw1,dw2
    
    def mlptrain(self,nIt):
        """ train the perceptron """
        
        #print('Training with',nIt,'Iterations')
        
        #First Output
        (self.o1,self.o2) = self.mlpfwd(self.x,False)
        #print('Start Error:',self.errfunc(self.o2,self.t))
            
        #for each iteration
        for n in range(nIt):
            
            if np.mod(n,1000)==0 and n!=0: 
                (self.o1,self.o2) = self.mlpfwd(self.x,False)
                print(n,'Error:',self.errfunc(self.o2,self.t))
            
            #randomize over input indiced for improved learning
            randind = np.arange(len(self.x)) 
            np.random.shuffle(randind)
            
            #sequential training (train for each input vector)
            if self.traintype == 'sequential':
                #for each entry of the input list adjust the weights (sequential training)
                for i in randind:
                    #calculate outputs of each layer
                    (self.o1[i],self.o2[i]) = self.mlpfwd(self.x[i],False)
               
                    #update weights
                    (dw1,dw2) = self.dw(self.x[i],self.o1[i],self.o2[i],self.t[i],self.w1,self.w2)
                    self.w1 += dw1
                    self.w2 += dw2
                
            #batch training (train all input vectors at once)        
            elif self.traintype == 'batch':
                (self.o1,self.o2) = self.mlpfwd(self.x,False)
                
                #udpate weights
                (dw1,dw2) = self.dw(self.x,self.o1,self.o2,self.t,self.w1,self.w2)
                self.w1 += dw1
                self.w2 += dw2
                #print(dw1,dw2)
                
        #Final output
        (self.o1,self.o2) = self.mlpfwd(self.x,False)
        #print('Final Error:',self.errfunc(self.o2,self.t))
              
        return 0
    
    def mlptrain_automatic(self,valid,validt,itSteps):
        """ train the perceptron until the error on the validation data increases """
        
        trainerror = np.array([])
        validerror = np.array([])

        n = -1    
        while n < 2 or (validerror[n] < validerror[n-1]):
            n+=1
            trainerror = np.append(trainerror,self.errfunc(self.mlpfwd(self.x,False)[1],self.t))
            validerror = np.append(validerror,self.errfunc(self.mlpfwd(valid,True)[1],validt))
            self.mlptrain(itSteps)
        
        return trainerror,validerror
        

    def confmat(self,inputs,targets):
        """ confusion matrix to evaluate the performance of the network """
        
        #outputs of perceptron
        (o1,o2) = self.mlpfwd(inputs,True)
        
        #output activations
        outputs = np.where(o2>0.5,1,0)
        
        classes = np.unique(targets,axis=0)
        nClasses = np.shape(classes)[0]
        cm = np.zeros((nClasses,nClasses))
        for i in range(nClasses):
            for j in range(nClasses):
                for n in range(np.shape(targets)[0]):
                    if (outputs[n]==classes[i]).all() and (targets[n]==classes[j]).all():
                        cm[i,j]+=1
        #print('targets\n',targets)
        #print('outputs\n',o2)
        #print('classes\n',classes)        
        print('Confusion Matrix\n',cm)
        if np.sum(cm)!=0:
            print('Succeed Rate',100*np.trace(cm)/np.sum(cm),'%')
        return 0
            








