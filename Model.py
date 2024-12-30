#We will use Count Vectorizer to convert the text data into numerical data that can be used to train the model
#Count vectorizer crates a matrix where the words are columns and the values in rows represent their frequency.

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from cleaningFunction import clean_and_lemmatize
from xgboost import XGBClassifier
import pandas as pd
import matplotlib.pyplot as plt



#This vectorizer will only include the most frequent 1500 words in the reviews
cv=CountVectorizer(max_features=1000)

ReviewsDf=pd.read_csv('CleanedLemmatizedReviews-*.csv')

score_counts = ReviewsDf['Score'].value_counts().sort_index()

plt.figure(figsize=(10, 6))
score_counts.plot(kind='bar')
plt.xlabel('Score')
plt.ylabel('Number of Reviews')
plt.title('Number of Reviews with Different Score Values')
plt.show()

ReviewsText=ReviewsDf['LemmatizedText']

X= cv.fit_transform(ReviewsText).toarray() #This will create a matrix in which each row is a review text and column contains 1500 words.
#Then the corresponding values represent the frequency of the word in the review text.

Y=ReviewsDf['Score']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


#now,lets try different types of model to check their accuracy for this dataset


#First logistics regression with softmax classifier
def LogisticsRegressionModel(x_train,y_train):

    model = LogisticRegression(multi_class='multinomial', solver='lbfgs',max_iter=2000)

    # Train the model
    model.fit(x_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(x_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')

    print('Classification Report:')
    print(classification_report(y_test, y_pred))

    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))
    return model


#Now Lets try XGBoost Classifier
def XGBoostModel(X,Y,x_test,y_test):
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss',n_estimators=400)

    Y=Y.apply(lambda x:x-1)

    xgb_model.fit(X,Y)

    # Make predictions on the test set using XGBoost
    y_pred_xgb = xgb_model.predict(x_test) + 1

    # Evaluate the XGBoost model
    accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
    print(f'XGBoost Accuracy: {accuracy_xgb:.2f}')

    print('XGBoost Classification Report:')
    print(classification_report(y_test, y_pred_xgb))

    print('XGBoost Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred_xgb))
    return xgb_model

def InputToPredict(model):
    review=str(input("Enter a review to predict the sentiment:"))
    review=clean_and_lemmatize(review)
    prediction=model.predict(cv.transform([review]).toarray())
    print(f'The sentiment of the review is: {prediction[0]}')

model=LogisticsRegressionModel(x_train,y_train)
InputToPredict(model)




#The accuracy is slightly misleading because the prediction is more likely to be around their correct value like 5 rating being around 4 or 5.
model=XGBoostModel(X=x_train,Y=y_train,x_test=x_test,y_test=y_test)

InputToPredict(model)
