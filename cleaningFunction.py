import spacy
import re
import pandas as pd
from spacy.lang.en.stop_words import STOP_WORDS

pattern= re.compile('[^a-zA-Z]') #This matches all the non-alphabetical characters

#We are trying to remove the stopwords from the review texts as they are not essential 
#But words like 'not' and 'no' are important in sentiment analysis as they differentiate between positive and negative review
words_to_remove={'not','no','nor','n\'t','neither','cannot','nobody','n`t','none','noone' }
STOP_WORDS.difference_update(words_to_remove)

#Now lets use stemming to reduce the words to their root form
#We can directly use lemmatization from spacy as it is better alternative to stemming
nlp = spacy.load('en_core_web_sm')
def clean_and_lemmatize(text):
    if pd.isna(text):
        return ""
    
    text=str(text)

    text = pattern.sub(' ', text).lower()
    #The words are auto tokenized and lemmatized with the help of spacy
    doc = nlp(text)
    # Remove stopwords and lemmatize the remaining words
    lemmatized_words = [token.lemma_ for token in doc if token.text not in STOP_WORDS]
    return ' '.join(lemmatized_words)

if __name__ == "__main__":
    # This code will only run when CleaningData.py is executed directly
    print("This is CleaningData.py being run directly")