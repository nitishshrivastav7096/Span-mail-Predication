import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer


def welcome():
    print("Welcome in Snap mail Predication system")
    print("Press ENTER key to proceed")
    input()

def checkcsv():
    csv_files=[]
    cur_dir=os.getcwd()
    content=os.listdir()
    for file_name in content:
        if file_name.split('.')[-1]=='csv':
            csv_files.append(file_name)
    return csv_files


def select_csv_file(csv_files):
    i=0
    for file_name in csv_files:
        print(i,'......',file_name)
        i+=1
    return csv_files[int(input("Select Your CSV file"))]    

def main():
    
    
 welcome()
 try:
    csv_files=checkcsv()
    csv_file=select_csv_file(csv_files)
    print(csv_file)
    print("Reading csv file......")
    print("Creating dataset....")
    raw_dataset=pd.read_csv(csv_file)

    #replace the mission values with null string
    dataset=raw_dataset.where((pd.notnull(raw_dataset)),'')
    print(dataset.head())
    print("Dataset is created")
    #label the span values as 0  and ham values as 1
    dataset.loc[dataset['Category']=='spam','Category',]=0
    dataset.loc[dataset['Category']=='ham','Category',]=1
    print(dataset.head())

    #separating the data as texts and labels
    x=dataset['Message']
    y=dataset['Category']
  
    #splitting data into training data and test data
    s=float(input("Enter test data size (between 0 and 1)--->"))
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=s,random_state=3)
    print("Creating ML model.......")

    #Transform the text data to feature vectors that can be used as a input data
    feature_extraction=TfidfVectorizer(min_df=1,stop_words='english',lowercase='true')
    x_train=feature_extraction.fit_transform(x_train)
    x_test=feature_extraction.transform(x_test)

    #covert y_train and y_test into integers
    y_train=y_train.astype('int')
    y_test=y_test.astype('int')

    
    #training model 
    model=LogisticRegression()
    model.fit(x_train,y_train)

    #evaluative of trained model
    y_train_predict=model.predict(x_train)
    y_accuracy=accuracy_score(y_train,y_train_predict)
    print("Training model accuracy is %2.2f%%"%(y_accuracy*100))

    print("Press ENTER key to know Model accuracy")
    input()

    #evaluative of test model
    y_predict=model.predict(x_test)
    accuracy=accuracy_score(y_test,y_predict)
    print("Our Model accuracy score is %2.2f%% "%(accuracy*100))

    print("ML model is created Now you can use")
    print("Press ENTER to input data")
    user=input()
    user_mail=[]
    user_mail.append(user)

    mail=feature_extraction.transform(user_mail)
    result=model.predict(mail)
    if result[0]==1:
        print("mail is ham")
    else:
        print("mail is spam")
 except FileNotFoundError:
     print("File is Not found")
     exit()
        
    
    


if __name__=="__main__":
    main()
    
