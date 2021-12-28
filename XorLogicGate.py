import numpy as np


x=np.array([[0,0,1,1],[0,1,0,1]])
y=np.array([[0,1,1,0]])
b=0.5
w=np.random.rand(2,1)
m = x.shape[1]
lossess=[]
learningRate=0.01