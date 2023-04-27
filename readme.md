# Amazon Reviews Sentiment Analysis

This code performs sentiment analysis on Amazon reviews using machine learning. The dataset used for training and testing the models is amazon_reviews.csv. The following models were trained and tested on the dataset:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Support Vector Machine
        
The model that achieved the highest accuracy was the Support Vector Machine (SVM) with an accuracy of 0.93. A grid search was then performed on the SVM model to find the best hyperparameters. 
The best parameters found were:

C: 1

dual: True

loss: squared_hinge

penalty: l2

The accuracy of the SVM model with the best hyperparameters was 0.94, which is only slightly higher than the accuracy of the original SVM model.

The load and preprocess the dataset section reads the amazon_reviews.csv file, drops any rows with missing data, shuffles the rows, and renames the columns.


The define functions for each model section defines four functions, each of which trains and tests a different machine learning model. The models are all trained using a TF-IDF vectorizer to convert the text reviews into numerical features that can be used as input to the models. The output of each function is the accuracy of the model on the test set.

The evaluate the performance of each model and choose the best one section evaluates the performance of each model by calling the functions defined in the previous section and printing the accuracy of each model. The best performing model is the SVM model.

The perform hyperparameter tuning on the best performing model section performs a grid search on the SVM model to find the best hyperparameters. The best hyperparameters are then used to train a new SVM model, and the accuracy of this model is printed.





