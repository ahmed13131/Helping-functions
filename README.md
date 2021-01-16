![1_eVZyKIcUXOfzPrMTGx7yVw](https://user-images.githubusercontent.com/59618586/104822204-81e41f00-5849-11eb-8b7b-2795c53cc2ae.png)
# Helping-functions (Demo):

## What it is about:
a Library to help u make work more easier to do more in less time by dealing with data and in the way you want

<br/>

## Existing functions:

#### p_val_of_features(df,label_column): 
calculate Ordinary Least Squares to know which features are important and make overview of the regression model.

* df -> the dataframe.
* label_column -> the column you want to predict.

<br/>

#### fill_lin_rand(messy_df, metric, colnames): 
fill the Nan values by Linear Regression or by Randon Forest.

* messy_df -> data frame you want to work on.
* metric -> model you want to work with (Linear Reg , Random forest).
* colnames-> columns with Nan vlaues.

<br/>

#### reduce_mem_usage(props):
reduce memory usage of the dataframe.

* props-> dataset you want to reduce

<br/>

#### dealing_with_outliers(df , type_o):
removes outliers with two techniques z-score and Inter-Quartile Range. 

* df -> dataframe
* type_o -> type of the method you want to choose

<br/>

#### log_features(x_train , x_test=None):
apply Log function to the features.

* x_train -> training data
* x_test -> testing data if your data is big enough

