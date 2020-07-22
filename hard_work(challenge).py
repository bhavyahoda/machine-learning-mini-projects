#run the full file block by block to get a better visulisation of what is happening actually
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


X=pd.read_csv('./training data/hard_work(challenge)/Linear_X_Train.csv')#always location is relative to the current directory
Y=pd.read_csv('./training data/hard_work(challenge)/Linear_Y_Train.csv')

#the above commands are to read the files relative to the current directory you are working in

Y.head()
X.head()
#to check if the data is correctly loaded or not 

plt.style.use("seaborn")#you can chose your own styling for the graphs
plt.scatter(X,Y,color="red")
plt.title("marks vs performance graph")
plt.xlabel("Hardwork")
plt.ylabel("performance")
plt.show()


type(X)
#so since X is in pandas right now so we convert it to numpy so that things are much more easier
#because it is best to work in numpy

x=X.values
y=Y.values
#we only normalise the x values and not the y values



u=x.mean()
std=x.std()
x=(x-u)/std
print(u,std)
#taken mean and standard deviation


#this is the hypothesis function
def hypothesis(x,theta):
    y_ = theta[0] + theta[1]*x
    return y_

#this function below is for finding the gradient
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


#this function below is just to calculate error and check if our error is really going down or not
def error(X,Y,theta):
    m = X.shape[0]
    total_error = 0.0
    for i in range(m):
        y_ = hypothesis(X[i],theta)
        total_error += (y_ - Y[i])**2
        
    return (total_error/m)


#this below function is the main gradient descent function 
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


#here this call is given to gradient descent function which gives call to both error and gradient functions and which in turn gives call to hypothesis function
theta,error_list,theta_list=gradientDescent(x,y)


plt.style.use("seaborn")#this is again just a style
plt.plot(error_list)#plotting the error list
plt.title("reduction of error over time")
plt.show()



y_=hypothesis(x,theta)
print(y_)



plt.scatter(x,y)
plt.plot(x,y_,color="cyan",label="prediction")
plt.legend()#to display the labels this legend function is used
plt.show()


x_test=pd.read_csv("./training data/hard_work(challenge)/Test Cases/Linear_X_Test.csv").values
y_test=hypothesis(x_test,theta)
y_test.shape


df=pd.DataFrame(data=y_test,columns=["y"])
df.to_csv("y_prediction.csv",index=False)
#this is the csv file which needs to be submitted 
#this csv file will be made inside the current directory which you are working in


#this is to calculate the accuracy 
#this is the formula used in online challenges also 
def r2_score(y,y_):
    num=np.sum((y-y_)**2)
    denom=np.sum((y-y.mean())**2)
    score=(1-num/denom)
    return score*100
s
r2_score(y,y_)


#this below save function is used to save the numpy file as Thetalist.npy 
theta_list
np.save("Thetalist.npy",theta_list)
