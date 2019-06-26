import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

yelp = pd.read_csv('yelp.csv')
yelp['text length'] = yelp['text'].apply(len) #create new column to record number of words in each review

# Exploring data
g = sns.FacetGrid(yelp,col='stars')
g.map(plt.hist, 'text length') # From histograms, we see that distribution of text length is similar across all star ratings

sns.boxplot(x='stars',y='text length',data=yelp) # "Text length" doesn't seem to be a helpful feature to build our classifier on, since there are so many outliers to account for
sns.countplot(x='stars',data=yelp)

df = yelp.groupby('stars')
new_df = df.mean()
new_df.corr()
sns.heatmap(new_df.corr(),cmap='coolwarm',annot=True) #Observe how different review ratings correlate with one another

# Classification
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

yelp_class = yelp[(yelp.stars == 1) | (yelp.stars == 5)] #Just look at 1 and 5-star reviews for simplicity

X = yelp_class['text']
Y = yelp_class['stars']
X = cv.fit_transform(X) #Vectorize X
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.3, random_state=101)

# Evaluate model
nb = MultinomialNB()
nb.fit(X_train,Y_train)
pred = nb.predict(X_test)
print(confusion_matrix(Y_test,pred))
print(classification_report(Y_test,pred)) #Model does well (precision and recall arounf 90%), given only 1 and 5-star data

# Create pipeline
pipeline = Pipeline([
    ('bow', CountVectorizer()),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])

X = yelp_class['text']
Y = yelp_class['stars']
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.3, random_state=101)
pipeline.fit(x_train,y_train)

#Evaluate pipeline
predictions = pipeline.predict(x_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions)) #Tfidf made the model worse
