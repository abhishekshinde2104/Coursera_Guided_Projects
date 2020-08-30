import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from utils import *
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, LambdaCallback

%matplotlib inline
#tf.logging.set_verbosity(tf.logging.ERROR)

print('Libraries imported.')

column_names=['serial','date','age','distance','stores','latitude','longitude','price']
df=pd.read_csv('data.csv',names=column_names)
df.head()

df.isna().sum()
#in practical world we can have missing values
#so its not possible to look at all the values
#hencewe take sum of all the columns after isna fucntion 


#data normalizatioin is done to bring everyhting to a scale
#1st column isnt useful so remove it
df=df.iloc[:,1:]
df_norm=(df-df.mean())/df.std()
df_norm.head()

y_mean=df['price'].mean()
y_std=df['price'].std()

def convert_label_value(pred):
    return int(pred*y_std+y_mean)

print(convert_label_value(0.350080))

x=df_norm.iloc[:,:6]#features of matrix
x.head()


y=df_norm.iloc[:,-1]
y.head()


x_arr=x.values
y_arr=y.values

print('Features array shape : ',x_arr.shape)
print('Label array shape : ',y_arr.shape)


x_train,x_test,y_train,y_test=train_test_split(x_arr,y_arr,test_size=0.05,random_state=0)


def get_model():
    model=Sequential([
        Dense(10,input_shape=(6,),activation='relu'),#1st hidden layer
        Dense(20,activation='relu'),#2nd hidden layer
        Dense(5,activation='relu'),#3rd hidden layer
        Dense(1)#ouput layer
        #since its a linear ouput wwe dont need any activation function here 
    ])
    
    model.compile(loss='mse',optimizer='adam')
    
    return model

get_model().summary()


es_cb=EarlyStopping(monitor='val_loss',patience=5)
#val_loss-->>test set
#patience is the no of epochs after which if the it should stop training if val_loss decreases

model=get_model()

preds_on_untrained=model.predict(x_test)

history=model.fit(
    x_train,y_train,
    validation_data=(x_test,y_test),
    epochs=100,
    callbacks=[es_cb]
)

#the earlystopping callback didnt see any improvement hence stops at 17th epoch

plot_loss(history)  



preds_on_trained=model.predict(x_test)
compare_predictions(preds_on_untrained,preds_on_trained,y_test)


price_untrained=[convert_label_value(y) for y in preds_on_untrained]
price_trained=[convert_label_value(y) for y in preds_on_trained]
price_test=[convert_label_value(y) for y in y_test]

compare_predictions(price_untrained,price_trained,price_test)





















