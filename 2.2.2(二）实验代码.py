import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
def Iris_type(s):
    it={b'Iris-setosa':0,b'Iris-versicolor':1,b'Iris-virginica':2}
    return it[s]


path="C:/Users/cs/Desktop/Iris.data.txt"
data=np.loadtxt(path,dtype=float,delimiter=',',converters={4:Iris_type})
x,y=np.array_split(data,(4,),axis=1)
x=x[:,:2]
logreg=LogisticRegression()
logreg.fit(x,y.ravel())
N,M=500,500
x1_min,x1_max=x[:,0].min(),x[:,0].max()
x2_min,x2_max=x[:,1].min(),x[:,1].max()
t1=np.linspace(x1_min,x1_max,N)
t2=np.linspace(x2_min,x2_max,M)
x1,x2=np.meshgrid(t1,t2)
x_test=np.stack((x1.flat,x2.flat),axis=1)

y_hat=logreg.predict(x_test)
y_hat=y_hat.reshape(x1.shape)
plt.pcolormesh(x1,x2,y_hat,cmap=plt.cm.prism)
plt.scatter(x[:,0],x[:,1],c=np.squeeze(y),edgecolors='k',cmap=plt.cm.prism)
plt.xlabel('Sepal lenth')
plt.ylabel('Sepal width')
plt.xlim(x1_min,x1_max)
plt.ylim(x2_min,x2_max)
plt.grid()
plt.show()
y_hat=logreg.predict(x)
y=y.reshape(-1)
print(y_hat.reshape)
print(y.shape)
result=y_hat==y
print(y_hat)
print(y)
print(result)
c=np.count_nonzero(result)
print(c)
print('Accuracy:%.2f%%'%(100*float(c)/float(len(result))))

