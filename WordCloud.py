import pandas  as pd
import matplotlib.pyplot as plt
import os

df=pd.read_csv('CleanedLemmatizedReviews-*.csv')

print(len(df))


#We now create a wordcloud to visualize the most common words for each classes of sentiments.
#For that we  use wordcloud module
#This creates a intuitive visual in which the size of a word is directly proportional to the frequency of that word
#We can create seperate wordclouds for different classes of score.

from wordcloud import WordCloud

wc=WordCloud(width=800,height=800,background_color='white',min_font_size=10)

#Now, we can create a wordcloud for each sentiment class
def WC_class(data,cls):
    #We will filter out the reviews with the given sentiment class
    data=data[data['Score']==cls]
    #We will combine all the reviews into a single string
    text=' '.join(data['LemmatizedText'])
    return wc.generate(text)

#Lets create wordclouds for each class and save the images to a folder
output_folder = 'wordclouds'
os.makedirs(output_folder, exist_ok=True)


for i in range(1,6):
    wordcloud=(WC_class(df,i))
    file_path = os.path.join(output_folder, f'wordcloud_score_{i}.png')
    wordcloud.to_file(file_path)
    print(f'Saved word cloud for score {i} to {file_path}')


#By observing the wordclouds, we can see that some words like 'bad', 'love', 'not', e.t.c can be used to differentiate between the classes
#Although, word like good and love are present in many classes, their frequency can be used to distinguish between the classes
#We will train a model which will be able to recognize these patterns
