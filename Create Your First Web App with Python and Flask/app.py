#print(__name__) #prints __main__
#this happens when the project runs successfully

from flask import Flask,render_template
from flask_sqlalchemy import SQLAlchemy

app=Flask(__name__) #here the app has been instantiated
#this app.py file doesnt have any knowledge about the route.py soo...

app.config['SECRET_KEY'] = '1234'
#this key  is used to generate the csrf token
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///data.db'

db=SQLAlchemy(app)

from routes import *


if __name__ == '__main__':
	app.run(debug=True)
