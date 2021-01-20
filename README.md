![1_eVZyKIcUXOfzPrMTGx7yVw](https://user-images.githubusercontent.com/59618586/104822204-81e41f00-5849-11eb-8b7b-2795c53cc2ae.png)
# Helping-functions for Data preprocessing (Demo):

## What it is about:
Functions to help you make work more easier to do more in less time by dealing with data and in the way you want

<br/>

## Existing functions:

#### p_val_of_features ( df , label_column ): 
calculate Ordinary Least Squares to know which features are important and make overview of the regression model.

* df -> the dataframe.
* label_column -> the Dependent column.

Returns : the summary of the independant features and its relation with the dependant feature

<br/>

#### fill_lin_rand ( messy_df, metric , colnames ): 
fill the Nan values by Linear Regression or by Random Forest.

* messy_df -> data frame you want to work on.
* metric -> model you want to work with (Linear Reg , Random forest).
* colnames-> columns with Nan vlaues.

Returns : the cleaned dataframe after filling the Nan values and a list of the missing values

<br/>

#### reduce_mem_usage ( props ):
reduce memory usage of the dataframe.

* props-> dataset you want to reduce

Returns : the reduced dataframe

<br/>

#### dealing_with_outliers( df , type_o = "z-score" ):
removes outliers with two techniques z-score and Inter-Quartile Range. 

* df -> dataframe
* type_o -> type of the method you want to choose

Returns : the dataframe after removing the outliers

<br/>

#### log_features ( x_train , x_test = None ):
apply Log function to the features.

* x_train -> training data
* x_test -> testing data 

Returns : x_train after applying the log and x_test after applying the log

## How to use it:
![Annotation 2021-01-21 013753](https://user-images.githubusercontent.com/59618586/105254033-5767ce00-5b89-11eb-8f59-d08b0eaf777f.png)
