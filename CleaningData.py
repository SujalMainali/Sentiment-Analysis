#We can perform this sentiment analysis by generating word clouds for each classes.
#Analyzing the most frequenctly occuring words in each sentiment class
#Maybe examine review length with the sentiment class

import numpy as np
import pandas as pd
import dask.dataframe as dd
from cleaningFunction import clean_and_lemmatize
import matplotlib.pyplot as plt

#This is the dataset cleaned from the Notebook of this project
df=pd.read_csv('ModifiedReviews.csv')
#print(df.head(5)) 





#Lets add the no. of characters in a new column in each review
df['Review Length'] = df['Text'].apply(len)


#lets create a column to store the no of words
df['Word Count'] = df['Text'].apply(lambda x: len(str(x).split()))


#Lets remove all the non-alphabetical characters from the reviews as they are just deadweights.
#For that lets write a regex
#We will clean all the non-alphabetical characters from the reviews, then split the review text into individual words
#We are also converting the entire text to lowercase
#This is the cleaning phase of any NLP project
#We are also removing the stopwords from the reviews with the help of spacy.




def clean_each_partition(partition):
    partition['LemmatizedText'] = partition['Text'].apply(clean_and_lemmatize)
    return partition



#We now convert the pandas dataframe to dask dataframe to speed up the processing
#This dask dataframe is a parallelized version of the pandas dataframe
#The pandas dataframe is split into multiple partitions and each partition is processed in parallel, we can specify the max no. of partitions.
#The partitions also helps in memeory management as we can easily run out of memory when processing the dataset but process small partitions doesnt overwhem memory.

ddf = dd.from_pandas(df, npartitions=20)



ddf = ddf.map_partitions(clean_each_partition)

# Write directly to CSV in chunks
ddf.to_csv('CleanedLemmatizedReviews-*.csv', index=False, single_file=True)














