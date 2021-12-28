import numpy as np
import matplotlib.pyplot as plt

x=np.array([[0,0,1,1],[0,1,0,1]])
y=np.array([[0,1,1,0]])
b=0.5
w=np.random.rand(2,1)
m = x.shape[1]
lossess=[]
learningRate=0.01

def sigmoid(z):
    z=1/(1+np.exp(-z))
    return z

def forwardProp(x,w,b):
    A = sigmoid(np.dot(x.T, w) + b)
    return A

def backwardProp(A):

    dz=(A.T-y)
    dw = 1. / m * np.dot(x, dz.T)
    db = 1. / m * np.sum(dz.T)

    return dw,db

for i in range(10000):
    A=forwardProp(x,w,b)
    cost = -1. / m * np.sum(y * np.log(A) + (1 - y) * np.log(1 - A))
    lossess.append(cost)
    dw,db=backwardProp(A)

    w=w-learningRate*dw
    b=b-learningRate*db

plt.plot(lossess)
plt.xlabel("EPOCHS")
plt.ylabel("Loss value")
plt.show()

def predict(w,b,input):
    A=forwardProp(test,w,b)
    A = np.squeeze(A)
    if A >= 0.5:
        print("For input", [i[0] for i in input], "output is 1")
    else:
        print("For input", [i[0] for i in input], "output is 0")


test =np.array([[1],[0]])
# print(test)
predict(w,b,test)


