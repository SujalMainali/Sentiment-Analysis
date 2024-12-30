This is a Sentiment Analysis Project Utilizing the Spacy framework for lemmitization.
In this project, I have utilized regex, spacy and pandas for cleaning the dataset.
Also, I have utilized Matplotlib for purpose of Visualizing the data along with checking imbalances in the data.
I also learned about wordclouds which helps in easily visualizing the Various classes of data. 
Similarly, We have utilized the dask library for processing very large datasets. We were unable to use just pandas because Pandas was not
originally built to scale. We used Dask to segment our full datasets into many workable size.
 
Finally, we used sklearn to build various models for classification. With the balanced data the output was more accurate.

############    RUNNING   #################
To run this project one can create a virtual environment using  ##  python -m venv ProjectEnv  ##
Then, use the requirements.txt to easily import all the libraries  ## pip install -r requirements.txt  ##

After this the project is good to go. Run all the files in order
1. ProjectNotebook.ipynb
2. cleaningFunction.py
3. CleaningData.py
4. WordCloud.py
5. Model.py


############# FILES  ################
For the csv file download from this link and save as Reviews.csv
"https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews"
