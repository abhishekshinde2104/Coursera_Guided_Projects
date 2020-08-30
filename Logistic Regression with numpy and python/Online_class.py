import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
plt.style.use("ggplot")
%matplotlib inline

from pylab import rcParams
rcParams['figure.figsize'] = 12, 8


"""Unlike in linear regression where the output is a continuous number
Logistic Regression is a classification Algorithm and it tries to predict a discrete set of class label for given i/p
"""
data = pd.read_csv('DMV_Written_Tests.csv')
data.head()


data.info()


score = data[['DMV_Test_1','DMV_Test_2']].values#independent variables
result = data['Results'].values#dependent variable


#making a scaterplot here
passed = (result == 1).reshape(100,1) #reshape to make it a vector of 100 rows and 1 column
failed = (result ==0).reshape(100,1)

#we will use seaborn to create scaterplot
ax=sns.scatterplot(x=score[passed[:,0],0],
                   y=score[passed[:,0],1],
                  marker='^',
                  color='green',
                  s=60)

sns.scatterplot(x=score[failed[:,0],0],
                   y=score[failed[:,0],1],
                  marker='X',
                  color='red',
                  s=60)

ax.set(xlabel='DMV Test 1 Scores',ylabel='DMV Test 2 scores')
ax.legend(['Passed','Failed'])
plt.show()


"""we are gonna put the input as z and if its > 0.5 then green or else red
"""
def logistic_function(x):
    return 1/(1+np.exp(-x))
logistic_function(0)


"""Now that we have sigmoid fucntion we can define the cost function/logistic function
we here need to minimize J(theto) as this the cost function
we will use gradient descent to minimize cost function
"""

def compute_cost(theta,x,y):
    #m->no of samples
    m=len(y)#m=100
    y_pred=logistic_function(np.dot(x,theta))
    error=(y*np.log(y_pred))+((1-y)*np.log(1-y_pred))
    cost=-1/m*sum(error)
    gradient=1/m*np.dot(x.transpose(),(y_pred-y))
    return cost[0],gradient


"""Before performing gradient descent we should do data standardization so that different scales of various features
dont adversely bias our models towards incorrect results"""

mean_scores=np.mean(score,axis=0)
std_scores=np.std(score,axis=0)
#standardisation step
score=(score-mean_scores)/std_scores

rows=score.shape[0]
cols=score.shape[1]

#feature matrix X
#we will also add a column of ones which is going to be the intercept term
X=np.append(np.ones((rows,1)),score,axis=1) #X=[100,3]
y=result.reshape(rows,1) #y=[100,1]

#Initialise theta values
theta_init=np.zeros((cols+1,1))
cost,gradient=compute_cost(theta_init,X,y)

print("Cost at initialization : ",cost)
print("Gradients at initialization : ",gradient)



def gradient_descent(x,y,theta,alpha,iterations):
    costs=[]
    for i in range(iterations):
        cost,gradient=compute_cost(theta,x,y)
        theta-=(alpha*gradient)
        costs.append(cost)
    return theta,costs

theta,costs=gradient_descent(X,y,theta_init,1,200)

print("Theta after running gradient descent : ",theta)
print("Resulting Cost : ",costs[-1])


plt.plot(costs)
plt.xlable=("iterations")
plt.ylabel=("$J(\Theta)$")
plt.title=("Values of Cost fucntion over Iterations of Gradient Descent")


ax=sns.scatterplot(x=X[passed[:,0],1],
                  y=X[passed[:,0],2],
                  marker='^',
                  color='green',
                  s=60)
sns.scatterplot(x=X[failed[:,0],1],
                  y=X[failed[:,0],2],
                  marker='X',
                  color='red',
                  s=60)
    
ax.legend(['Passed','Failed'])
ax.set(xlabel='DMV Test 1 scores',ylabel='DMV Test 2 scores')

x_boundary=np.array([np.min(X[:,1]),np.max(X[:,1])])
y_boundary=-(theta[0]+theta[1]*x_boundary)/theta[2]

sns.lineplot(x=x_boundary,y=y_boundary,color='blue')
plt.show()


def predict(theta,x):
    result=x.dot(theta)
    return result > 0

p=predict(theta,X)
print("Training Accuracy : ",sum(p==y)[0],"%")


test=np.array([50,79])
test=(test-mean_scores)/std_scores
test=np.append(np.ones(1),test)
probability=logistic_function(test.dot(theta))
print("Person who scores 50 and 79 on there DMV Written test have a ",
      np.round(probability[0],2)," probability of passing")




