import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


X=pd.read_csv('./training data/hard_work(challenge)/Linear_X_Train.csv')#always location is relative to the current directory
Y=pd.read_csv('./training data/hard_work(challenge)/Linear_Y_Train.csv')




Y.head()
X.head()
#to check if the data is correctly loaded or not 
plt.style.use("seaborn")
plt.scatter(X,Y,color="red")
plt.title("marks vs performance graph")
plt.xlabel("Hardwork")
plt.ylabel("performance")
plt.show()


# In[4]:


type(X)
#so since X is in pandas right now so we convert it to numpy so that things are much more easier


# In[5]:


x=X.values
y=Y.values
#we only normalise the x values and not the y values


# In[6]:


u=x.mean()
std=x.std()
x=(x-u)/std
print(u,std)


# In[7]:


def hypothesis(x,theta):
    y_ = theta[0] + theta[1]*x
    return y_
def gradient(X,Y,theta):
    m = X.shape[0]
    grad = np.zeros((2,))
    for i in range(m):
        x = X[i]
        y_ = hypothesis(x,theta)
        y = Y[i]
        grad[0] += (y_ - y)
        grad[1] += (y_ - y)*x
    return grad/m

def error(X,Y,theta):
    m = X.shape[0]
    total_error = 0.0
    for i in range(m):
        y_ = hypothesis(X[i],theta)
        total_error += (y_ - Y[i])**2
        
    return (total_error/m)
def gradientDescent(X,Y,max_steps=100,learning_rate =0.1):
    theta = np.zeros((2,))
    error_list = []
    theta_list = []
    for i in range(max_steps):
        # Compute grad
        grad = gradient(X,Y,theta)
        e = error(X,Y,theta)[0]
        #Update theta
        theta[0] = theta[0] - learning_rate*grad[0]
        theta[1] = theta[1] - learning_rate*grad[1]
        # Storing the theta values during updates
        theta_list.append((theta[0],theta[1]))
        error_list.append(e)
    return theta,error_list,theta_list


# In[8]:


theta,error_list,theta_list=gradientDescent(x,y)


# In[9]:


theta


# In[10]:


plt.style.use("seaborn")
plt.plot(error_list)
plt.title("reduction of error over time")
plt.show()


# In[11]:


y_=hypothesis(x,theta)
print(y_)


# In[12]:


plt.scatter(x,y)
plt.plot(x,y_,color="cyan",label="prediction")
plt.legend()
plt.show()


# In[13]:


x_test=pd.read_csv("./training data/hard_work(challenge)/Test Cases/Linear_X_Test.csv").values
y_test=hypothesis(x_test,theta)
y_test.shape


# In[14]:


df=pd.DataFrame(data=y_test,columns=["y"])
df.to_csv("y_prediction.csv",index=False)


# In[15]:


def r2_score(y,y_):
    num=np.sum((y-y_)**2)
    denom=np.sum((y-y.mean())**2)
    score=(1-num/denom)
    return score*100


# In[16]:


r2_score(y,y_)
#this is how accuracy is determined in online competitions 
#see formula above


# In[17]:


theta_list
np.save("Thetalist.npy",theta_list)


# In[ ]:




