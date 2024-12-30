#We will use Count Vectorizer to convert the text data into numerical data that can be used to train the model
#Count vectorizer crates a matrix where the words are columns and the values in rows represent their frequency.

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from cleaningFunction import clean_and_lemmatize
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
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

#This is the function to Predict data
def InputToPredict(model):
    review=str(input("Enter a review to predict the sentiment:"))
    review=clean_and_lemmatize(review)
    prediction=model.predict(cv.transform([review]).toarray())
    print(f'The sentiment of the review is: {prediction[0]}')
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


# model=LogisticsRegressionModel(x_train,y_train)
# InputToPredict(model)





#The accuracy is slightly misleading because the prediction is more likely to be around their correct value like 5 rating being around 4 or 5.
model=XGBoostModel(X=x_train,Y=y_train,x_test=x_test,y_test=y_test)
InputToPredict(model)

#Now lets try using Neural Network to predict the sentiment

def NeuralNetworkModel(X, Y, x_test, y_test):
    # Prepare the dataset with batching
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.batch(32)  # Load in batches of 32

    # Prepare the test dataset as well
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.batch(32)

    # Define the Neural Network model
    Network = Sequential(
        [
            Dense(100, activation='relu', input_shape=(1000,)),  # Input shape: 1000 features
            Dense(50, activation='relu'),
            Dense(5, activation='softmax'),
        ]
    )

    Network.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    Network.fit(dataset, epochs=10, validation_data=test_dataset)

    # Evaluate the model
    loss, accuracy = Network.evaluate(test_dataset)
    print(f'Neural Network Accuracy: {accuracy:.2f}')

    # Make predictions on the test set
    y_pred_nn = Network.predict(x_test)
    y_pred_nn_classes = y_pred_nn.argmax(axis=-1)

    # Evaluate the Neural Network model
    print('Neural Network Classification Report:')
    print(classification_report(y_test, y_pred_nn_classes))

    print('Neural Network Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred_nn_classes))

    return Network

# Example call to the function
#The accuracy of this Neural Network is less than the XGBoost so the XGBoost is the best model for this dataset.

# Network = NeuralNetworkModel(X=x_train, Y=y_train, x_test=x_test, y_test=y_test)
# InputToPredict(Network)


#This is the function to save effective models
def save_model(model, filename):
    model.save(filename)
    print(f'Model saved to {filename}')

