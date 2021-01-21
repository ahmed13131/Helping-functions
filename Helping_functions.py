import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

import statsmodels.formula.api as sm
from scipy import stats


def p_val_of_features(df,label_column):
    
    """
    this function calculate the P value of the features to know how it affects the regression module
    for a single label
    
    df -> the dataframe
    label_column -> the column you want to predict
    
    """
    # stre is a string variable of independent and dependant columns
    
    stre = '{} ~'.format(label_column) #the format type that ols model accept 
    
    for i in df.columns:
        stre = stre + "{} +".format(i)
        
    stre = stre[0:-1]  #to remove the last + sign
    
    reg_ols = sm.ols(formula=stre, data=df).fit() 
    
    return reg_ols.summary()



def fill_lin_rand(messy_df, metric, colnames):
    """
    this function is to fill the Nan values of columns by making 
    a regression modle by taking features of Non-Nan values as a
    training data and predicting the missing values and fill it
    
    messy_df -> data frame you want to work on
    metric -> model you want to work with (Linear Reg , Random forest)
    colnames-> columns with Nan vlaues
    
    """
    
    # Create X_df of predictor columns
    X_df = messy_df.drop(colnames, axis = 1)
    
    # Create Y_df of predicted columns
    Y_df = messy_df[colnames]
        
    # Create empty dataframes and list
    Y_pred_df = pd.DataFrame(columns=colnames)
    Y_missing_df = pd.DataFrame(columns=colnames)
    missing_list = []
    
    # Loop through all columns containing missing values
    for col in messy_df[colnames]:
    
        # Number of missing values in the column
        missing_count = messy_df[col].isnull().sum()
        
        # Separate train dataset which does not contain missing values
        messy_df_train = messy_df[~messy_df[col].isnull()]
        
        # Create X and Y within train dataset
        msg_cols_train_df = messy_df_train[col]
        messy_df_train = messy_df_train.drop(colnames, axis = 1)

        # Create test dataset, containing missing values in Y    
        messy_df_test = messy_df[messy_df[col].isnull()]
        
        # Separate X and Y in test dataset
        msg_cols_test_df = messy_df_test[col]
        messy_df_test = messy_df_test.drop(colnames,axis = 1)

        # Copy X_train and Y_train
        Y_train = msg_cols_train_df.copy()
        X_train = messy_df_train.copy()
        
        # Linear Regression model
        if metric == "Linear Regression":
            model = LinearRegression()
            model.fit(X_train,Y_train)
            print("R-squared value is: " + str(model.score(X_train, Y_train)))
          
        # Random Forests regression model
        elif metric == "Random Forests":
            model = RandomForestRegressor(n_estimators = 10 , oob_score = True)
            model.fit(X_train,Y_train) 
            
#             importances = model.feature_importances_
#             indices = np.argsort(importances)
#             features = X_train.columns
            
#             print("Missing values in"+ col)
#             #plt.title('Feature Importances')
#             plt.barh(range(len(indices)), importances[indices], color='b', align='center')
#             plt.yticks(range(len(indices)), features) ## removed [indices]
#             plt.xlabel('Relative Importance')
#             plt.show()
        
        X_test = messy_df_test.copy()
        
        # Predict Y_test values by passing X_test as input to the model
        Y_test = model.predict(X_test)
        
        Y_test_integer = pd.to_numeric(pd.Series(Y_test),downcast='integer')
        
        # Append predicted Y values to known Y values
        Y_complete = Y_train.append(Y_test_integer)
        Y_complete = Y_complete.reset_index(drop = True)
        
        # Update list of missing values
        missing_list.append(Y_test.tolist())
        
        Y_pred_df[col] = Y_complete
        Y_pred_df = Y_pred_df.reset_index(drop = True)
    
    # Create cleaned up dataframe
    clean_df = X_df.join(Y_pred_df)
    
    return clean_df,missing_list



def reduce_mem_usage(props):
    
    """
    this funaction to reduce memory usage of dataset
    
    props-> dataset you want to reduce
    
    """
    
    start_mem_usg = props.memory_usage().sum() / 1024**2
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in. 
    for col in props.columns:
        if props[col].dtype not in [object, bool]:  # Exclude strings
            
            # Print current column type
            print("******************************")
            print("Column: ",col)
            print("dtype before: ",props[col].dtype)
            
            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            mn = props[col].min()
            '''
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(props[col]).all(): 
                NAlist.append(col)
                props[col].fillna(mn-1,inplace=True) 
            '''
                   
            # test if column can be converted to an integer
            asint = props[col].fillna(0).astype(np.int64)
            result = (props[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True
            

            
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)    
            
            # Make float datatypes 32 bit
            else:
                props[col] = props[col].astype(np.float32)
            
            # Print new column type
            print("dtype after: ",props[col].dtype)
            print("******************************")
    
    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return props, NAlist



def dealing_with_outliers(df , type_o = "z-score"):
    
    """
    this function removes outliers with z-score and Inter-Quartile Range
    method
    
    hint : XGboost deal with it very good (SOTA machine learning model)
    
    df -> dataframe
    type_o -> type of the method you want to choose
    """
    
    if type_o == "z-score":
        
        #Now lets transform the dataframe by making the mean = 0 & std = 1.
        
        z = np.abs(stats.zscore(df))
        df = df[(z < 3).all(axis=1)]
        
    if type_o == "IQR" :
        
        #Inter-Quartile Range
        
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
        
    return df



def log_features(x_train , x_test=None):
    
    """
    log the data to remove large gaps between the data
    after or before removing outliers
    
    x_train -> training data
    x_test -> testing data if your data is big enough
    """
    
    x_log= np.log(x_train)
    x_log[x_log == -inf] = 0
    
    xt_log= np.log(x_test)
    xt_log[xt_log == -inf] = 0
    
    return x_log,xt_log





