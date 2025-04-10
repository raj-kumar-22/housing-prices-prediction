import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler
import scipy.stats
import math
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

# pd.set_option('max_columns', None) # only uncomment while testing and debugging
# pd.set_option('max_rows', None) # only uncomment while testing and debugging

# =============================================================================
def main():
    # Read the original data files
    trainDF = pd.read_csv("data/train.csv")
    testDF = pd.read_csv("data/test.csv")

    # demonstrateHelpers(trainDF, testDF)
    
    trainInput, testInput, trainOutput, testIDs = transformData(trainDF, testDF)
    doExperiment(trainInput, trainOutput)
    
    doKaggleTest(trainInput, testInput, trainOutput, testIDs)

    
# ===============================================================================
'''
Does k-fold CV on the Kaggle training set using LinearRegression.
(You might review the discussion in hw09 about the so-called "Kaggle training set"
versus other sets.)
'''
def doExperiment(trainInput, trainOutput):
    
    alg = GradientBoostingRegressor()
    cvScores = np.exp(np.sqrt(-model_selection.cross_val_score(alg, trainInput, trainOutput, scoring='neg_mean_squared_error', cv=10)))
    print("CV Average Score GBR:", cvScores.mean())
    
    alg = XGBRegressor()
    result = np.exp(np.sqrt(-model_selection.cross_val_score(alg, trainInput, trainOutput, scoring='neg_mean_squared_error', cv=10)))
    print("CV Average Score XGB:", np.mean(result))
    
    alg = RandomForestRegressor()
    result = np.exp(np.sqrt(-model_selection.cross_val_score(alg, trainInput, trainOutput, scoring='neg_mean_squared_error', cv=10)))
    print("CV Average Score Random Forest Regressor:", np.mean(result))
    
    #minimum MSE achieved using GBR
    
    
# ===============================================================================
'''
Runs the algorithm on the testing set and writes the results to a csv file.
'''
def doKaggleTest(trainInput, testInput, trainOutput, testIDs):
    alg = GradientBoostingRegressor()

    # Train the algorithm using all the training data
    alg.fit(trainInput, trainOutput)

    # Make predictions on the test set.
    predictions = np.exp(alg.predict(testInput)) 
    # predictions will be log of saleprice so need to be converted back
    

    # Create a new dataframe with only the columns Kaggle wants from the dataset.
    submission = pd.DataFrame({
        "Id": testIDs,
        "SalePrice": predictions
    })

    # Prepare CSV
    submission.to_csv('data/testrkk.csv', index=False)
    # Now, this .csv file can be uploaded to Kaggle
    # Kaggle Score: 0.13526
    
# ============================================================================
# Data cleaning - conversion, normalization

'''
Pre-processing code will go in this function (and helper functions you call from here).
'''
def transformData(trainDF, testDF):
    
    # checking number of missing values in each column (only uncomment following 2 lines while testing)
    # print('Attributes in TrainDF with missing values: \n', getAttrsWithMissingValues(trainDF))
    # print(trainDF.isna().sum())
    ''' 
    Findings of line number 77-78; colNames and number of missing values
    ['LotFrontage' (259), 'Alley'(1369), 'MasVnrType' and 'MasVnrArea'(8), 'BsmtQual', 'BsmtCond' and 'BsmtFinType1'(37), 'BsmtExposure'(38) and 'BsmtFinType2'(38), 'Electrical'(1), 'FireplaceQu'(690), 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', and 'GarageCond'(81), 'PoolQC'(1453), 'Fence'(1179), 'MiscFeature'(1406)]
    '''
    
    testIDs = testDF.loc[:, 'Id']
    trainOutputSeries = trainDF.loc[:, 'SalePrice']
    trainDF = trainDF.drop(['Id', 'SalePrice'], axis=1)
    testDF = testDF.drop(['Id'], axis=1)
    
    # Fixing data types in the raw data
    ''' 
    Upon examining the data_description.txt file, it became apparent that some numeric attributes actually
    have categorical ranges such as 'MSSubClass' which has 16 types/categories of dwellings 
    MSSubClass: Identifies the type of dwelling involved in the sale.	
        20	1-STORY 1946 & NEWER ALL STYLES; 30	1-STORY 1945 & OLDER; ...; 180 PUD - MULTILEVEL - INCL SPLIT LEV/FOYER, 190	2 FAMILY CONVERSION - ALL STYLES AND AGES
       This column clearly should be categorical
    '''
    trainDF.loc[:,'MSSubClass'] = trainDF.loc[:,'MSSubClass'].astype(str)
    testDF.loc[:,'MSSubClass'] = testDF.loc[:,'MSSubClass'].astype(str)
    
    # 
    '''
    Filling Missing Values (a) Non-Numeric Attributes
    
    While examining the data_description.txt file, I noticed that certain missing values actually communicated
    meaningful information. For example, Alley; Grvl = Gravel, Pave = Paved, NA = No alley access
    So filling Non-Numeric Missing Values isn't as simple as replacing NaN values with mode.
    '''
    # print('Only Non-Numeric Attributes with missing values')
    # print(trainDF.loc[:,getNonNumericAttrs(trainDF)].loc[:, trainDF.isna().sum() > 0].columns)
    # print(testDF.loc[:,getNonNumericAttrs(testDF)].loc[:, testDF.isna().sum() > 0].columns)
    '''Only Non-Numeric Attributes with missing values
    ['MSZoning','Alley','Utilities','Exterior1st','Exterior2nd','MasVnrType','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinSF1','BsmtFinType2','Electrical','KitchenQual', 'Functional','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PoolQC','Fence','MiscFeature','SaleType']
    For each column name in the list above, we manually check data_description.txt for any meaningful missing values
    The following columns were found to have meaningful information contained in 'missing' values:
    '''
    cols_with_meaningful_NaN = ['Alley','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
                                'FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PoolQC',
                                'Fence','MiscFeature']
    
    cols_with_no_meaningful_NaN = ['MSZoning','Utilities','Exterior1st','Exterior2nd','MasVnrType','Electrical',
                                      'KitchenQual','Functional','SaleType']
    
    # print(trainDF.isna().sum().sum() + testDF.isna().sum().sum()) #before this transformation total missing values = 13965
    for col_name in cols_with_meaningful_NaN:
        trainDF.loc[:,col_name] = trainDF.loc[:,col_name].fillna("NotNaN") #replace with string
        testDF.loc[:,col_name] = testDF.loc[:,col_name].fillna("NotNaN") #replace with string
        
    for col_name in cols_with_no_meaningful_NaN:
        trainDF.loc[:,col_name] = trainDF.loc[:,col_name].fillna(trainDF.loc[:,col_name].mode().iloc[0])
        testDF.loc[:,col_name] = testDF.loc[:,col_name].fillna(trainDF.loc[:,col_name].mode().iloc[0])
    # print(trainDF.isna().sum().sum() + testDF.isna().sum().sum()) #after this transformation total missing values = 678
    
    '''Filling Missing Values (b) Numeric Attributes'''
    
    # Finding out which Numeric Cols have missing values
    # print('NaN cols in trainDF:\n', getAttrsWithMissingValues(trainDF, True), end = '\n\n')
    # print('NaN cols in testDF:\n', getAttrsWithMissingValues(testDF, True))
    numeric_cols_with_NaN = ['LotFrontage', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 
                             'BsmtFullBath', 'BsmtHalfBath', 'GarageYrBlt', 'GarageCars', 'GarageArea']
    
    # print(trainDF.isna().sum().sum() + testDF.isna().sum().sum()) # before filling missing numeric values total missing values = 678
    for col_name in numeric_cols_with_NaN:
        # filling in missing numeric values using KNNRegressor
        # BEGIN: from https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html
        trainDF, testDF = knn_fillNaN(trainDF, testDF, col_name)
        # END: from https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html
    
    # print(trainDF.isna().sum().sum() + testDF.isna().sum().sum()) # after filling missing numeric values total missing values = 0
    
    '''
    Standardization (Scaling)
    '''
    skewDF = pd.DataFrame(getNumericAttrs(trainDF), columns=['Attribute'])
    skewDF['Skew'] = skewDF.loc[:,'Attribute'].map(lambda attr: scipy.stats.skew(trainDF.loc[:,attr])) # add a column called skew
    skewDF['Skewed'] = skewDF.loc[:,'Skew'].map(lambda skew: True if abs(skew) >= 0.3 else False)
    # print(skewDF)
    # Scaling the data using PowerTransformer
    # BEGIN: from https://towardsdatascience.com/how-to-differentiate-between-scaling-normalization-and-log-transformations-69873d365a94
    # EXPLANATION: When a feature does not follow a linear distribution, it would be unwise to use the mean and the 
    # standard deviation to scale it. The fact that some features will still be skewed after simple standardization 
    # confirms that standardization does not work on them.
    # To implement non-linear transformations, Sklearn offers a PowerTransformer class (uses logarithmic functions 
    # under the hood) that helps minimize skewness and map any distribution to a normal one as close as possible:
    pt = PowerTransformer()
    skewed_col_names = skewDF.loc[:,'Attribute'].loc[skewDF.loc[:,'Skewed'] == True]
    
    trainDF.loc[:,skewed_col_names] = pd.DataFrame(pt.fit_transform(trainDF.loc[:,skewed_col_names]), columns = skewed_col_names)
    testDF.loc[:,skewed_col_names] = pd.DataFrame(pt.fit_transform(testDF.loc[:,skewed_col_names]), columns = skewed_col_names)
    trainOutputSeries = np.log(trainOutputSeries) 
    # After predictions from the model, we will undo the log transform before submitting predictions
    
    # END: from https://towardsdatascience.com/how-to-differentiate-between-scaling-normalization-and-log-transformations-69873d365a94
    #print()
    #print(trainDF, testDF, sep ='\n\n')
    #print(skewDF)
    
    # Transforming cyclical or time attributes using Trignometric Functions
    # BEGIN: from https://towardsdatascience.com/cyclical-features-encoding-its-about-time-ce23581845ca
    # EXPLANATION: There is an attribute in the data_description.txt called 'MoSold.' Any algorithm that we try to fit 
    # to our data would think that MoSold is a number, or that higher or lower values actually do mean something. Well, 
    # they do. But, since, the values range from 1 to 12, it means that they are cyclical values based on time. So in 
    # order for any model to capture the information this attribute actually represents, it needs to be converted 
    # to cyclical values.
    # To transform cyclical features, we would apply a cosine function to all values in the 'MoSold' column
    # One thing to keep in mind is the period in Radian = 2 * pi
    #print(trainDF.loc[:,'MoSold'], sep ='\n\n')
    x_norm = 2 * math.pi * trainDF.loc[:,'MoSold'] / trainDF.loc[:,'MoSold'].max()
    trainDF.loc[:,'MoSold'] = np.cos(x_norm)
    testDF.loc[:,'MoSold'] = np.cos(x_norm)
    #print(trainDF.loc[:,'MoSold'], sep ='\n\n')
    # END: from https://towardsdatascience.com/cyclical-features-encoding-its-about-time-ce23581845ca
    
    '''
    One-hot Encoding of Non-Numeric Variables
    
    Due to disparities in some Non-Numeric columns in the test and train sets, for example, in the 
    number of categories certain categorical attributes have in the train set exceeds the number of
    categories for the same attributes in the test set (Utilities; 'AllPub', 'NoSeWa' in trainDF but 
    only 'AllPub' in testDF) and vice versa. So, it would be hard to maintain a consistent number of
    columns in both the sets after one-hot encoding.
    
    While I am aware of the philosophical difference between the train and test set, it might be more 
    orderly to merge the two sets (excluding the SalePrice column in the train set), and then spliting the 
    sets to hide the test set from the train set.
    I will then fit the model on the train set and predict the output for the test set.
    '''
    #print('trainDF\n', trainDF.shape)
    #print('testDF\n', testDF.shape)
    combinedDF = pd.concat([trainDF, testDF], axis = 0)   # Combine Train and Test Set
    combinedDF = pd.get_dummies(combinedDF)               # One-hot Encoding
    #print('dummiescombinedDF\n', combinedDF.shape)        
    trainDF = combinedDF.iloc[:1460, :]                   # Split Again into Train Set        
    testDF = combinedDF.iloc[1460:, :]                    # and Test Set
    # print('trainDF\n', trainDF.shape)
    # print('testDF\n', testDF.shape)
    
    '''
    Standardize Test and Train Set
    Mean = 0, Standard Deviation = 1
    '''
    # i wanted to standardize after one hot encoding as well
    # the standardization method below was giving missing values for a huge fraction of the dummy columns
    # print('NaN cols in trainDF:\n', getAttrsWithMissingValues(trainDF, True), end = '\n\n')
    # print('NaN cols in testDF:\n', getAttrsWithMissingValues(testDF, True))
    # trainDF.loc[:,:] = (trainDF.loc[:, :] - trainDF.loc[:, :].mean())/trainDF.loc[:, :].std()
    # testDF.loc[:,:] = (testDF.loc[:, :] - testDF.loc[:, :].mean())/testDF.loc[:, :].std()
    # print('NaN cols in trainDF:\n', getAttrsWithMissingValues(trainDF, True), end = '\n\n')
    # print('NaN cols in testDF:\n', getAttrsWithMissingValues(testDF, True))
    
    
    # So a built-in scikit preprocessing for standardization submodule is the best bet to avoid that.
    # BEGIN: from https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    # EXPLANATION: Standardize features by removing the mean and scaling to unit variance.
    # The standard score of a sample x is calculated as:
    # z = (x - u) / s
    std_scaler = StandardScaler()
    std_scaler.fit(trainDF)
    trainDF = pd.DataFrame(std_scaler.transform(trainDF), index=trainDF.index, columns=trainDF.columns)
    std_scaler.fit(testDF)
    testDF = pd.DataFrame(std_scaler.transform(testDF), index=testDF.index, columns=testDF.columns)
    # END: from https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    # print('NaN cols in trainDF:\n', getAttrsWithMissingValues(trainDF, True), end = '\n\n')
    # print('NaN cols in testDF:\n', getAttrsWithMissingValues(testDF, True))
    # this time, there are no columns with missing values in either the test or train set
    
    return trainDF, testDF, trainOutputSeries, testIDs
       
    #predictors = 1#['1stFlrSF', '2ndFlrSF']
    '''
    You'll want to use far more predictors than just these two columns, of course. But when you add
    more, you'll need to do things like handle missing values and convert non-numeric to numeric.
    Other preprocessing steps would likely be wise too, like standardization, get_dummies, 
    or converting or creating attributes based on your intuition about what's relevant in housing prices.
    '''
    
    #trainInput = 1#trainDF.loc[:, predictors]
    #testInput = 1#testDF.loc[:, predictors]
    '''
    Any transformations you do on the trainInput will need to be done on the
    testInput the same way. (For example, using the exact same min and max, if
    you're doing normalization.)
    '''
    
    #trainOutput = 1#trainDF.loc[:, 'SalePrice']
    #testIDs = 1#testDF.loc[:, 'Id']
    
    #return trainInput, testInput, trainOutput, testIDs, predictors
    

# ===============================================================================
'''
Fills missing values in numeric Cols using available values (of course, in the same column) of KNearestNeighbours
'''
def knn_fillNaN(trainDF, testDF, col_with_na_vals):
    # print(col_with_na_vals)
    numeric_trainDF = trainDF.loc[:, getNumericAttrs(trainDF)]
    complete_trainDF_columns = numeric_trainDF.loc[:, getAttrsWithMissingValues(numeric_trainDF, False)].columns
    numeric_testDF = testDF.loc[:, getNumericAttrs(testDF)]
    complete_testDF_columns = numeric_testDF.loc[:, getAttrsWithMissingValues(numeric_testDF, False)].columns
    
    y_train = numeric_trainDF.loc[numeric_trainDF.loc[:numeric_trainDF.shape[0],col_with_na_vals].isna() == False, col_with_na_vals] #series
    X_train = numeric_trainDF.loc[numeric_trainDF.loc[:numeric_trainDF.shape[0],col_with_na_vals].isna() == False, complete_testDF_columns] #df
    X_test_for_testDF = numeric_testDF.loc[numeric_testDF.loc[:,col_with_na_vals].isna() == True, complete_testDF_columns] #df
    # print('y_train shape', len(y_train))
    # print('X_train shape', X_train.shape)
    # print('X_test_for_testDF shape', X_test_for_testDF.shape)  
    '''
    The above commented lines are to ensure that train and test have same number of columns.
    The following if condition is there for a reason.
    as it turns out, all the numeric columns in the trainDF that have NaN Values (LotFrontage, MasVnrArea, 
    GarageYrBlt) also have NaN values in testDF. But not all numeric testDF columns with NaN values have 
    NaN values in the trainDF. Uncomment the following lines for proof! and to find out which Numeric Cols 
    in each set have missing values.
    '''
    # print('NaN cols in trainDF:\n', getAttrsWithMissingValues(trainDF, True), end = '\n\n')
    # print('NaN cols in testDF:\n', getAttrsWithMissingValues(testDF, True))
    
    knn = KNeighborsRegressor()
    if trainDF.loc[:,col_with_na_vals].isna().sum() > 0:
        X_train2 = numeric_trainDF.loc[numeric_trainDF.loc[:numeric_trainDF.shape[0],col_with_na_vals].isna() == False, complete_trainDF_columns] #df
        knn.fit(X_train2, y_train)
        X_test_for_trainDF = numeric_trainDF.loc[numeric_trainDF.loc[:,col_with_na_vals].isna() == True, complete_trainDF_columns] #df
        # print('X_train2 shape', X_train2.shape)
        # print('X_test_for_trainDF shape', X_test_for_trainDF.shape)
        predicted_values_for_trainDF = knn.predict(X_test_for_trainDF)
        trainDF.loc[trainDF.loc[:,col_with_na_vals].isna() == True, col_with_na_vals] = predicted_values_for_trainDF
    knn.fit(X_train, y_train)
    predicted_values_for_testDF = knn.predict(X_test_for_testDF)  
    testDF.loc[testDF.loc[:,col_with_na_vals].isna() == True, col_with_na_vals] = predicted_values_for_testDF
    
    return trainDF, testDF
# ===============================================================================
'''
Demonstrates some provided helper functions that you might find useful.
'''
def demonstrateHelpers(trainDF, testDF):
    print("Attributes with missing values:", getAttrsWithMissingValues(trainDF), sep='\n')
    
    numericAttrs = getNumericAttrs(trainDF)
    print("Numeric attributes:", numericAttrs, sep='\n')
    
    nonnumericAttrs = getNonNumericAttrs(trainDF)
    print("Non-numeric attributes:", nonnumericAttrs, sep='\n')
    print('\n\n\n')
    print("Values, for each non-numeric attribute in TrainDF:", getAttrToValuesDictionary(trainDF.loc[:, nonnumericAttrs]), sep='\n')
    print('\n\n\n')
    print("Values, for each non-numeric attribute in TestDF:", getAttrToValuesDictionary(testDF.loc[:, nonnumericAttrs]), sep='\n')

# ===============================================================================
'''
Returns a dictionary mapping an attribute to the array of values for that attribute.
'''
def getAttrToValuesDictionary(df):
    attrToValues = {}
    for attr in df.columns.values:
        attrToValues[attr] = df.loc[:, attr].unique()

    return attrToValues

# ===============================================================================
'''
Returns the attributes with missing values.
'''
def getAttrsWithMissingValues(df, boolean):
    valueCountSeries = df.count(axis=0)  # 0 to count down the rows
    numCases = df.shape[0]  # Number of examples - number of rows in the data frame
    missingSeries = (numCases - valueCountSeries)  # A Series showing the number of missing values, for each attribute
    if boolean:
        attsOfInterest = missingSeries[missingSeries != 0].index
    else:
        attsOfInterest = missingSeries[missingSeries == 0].index
    return attsOfInterest

# =============================================================================

'''
Returns the numeric attributes.
'''
def getNumericAttrs(df):
    return __getNumericHelper(df, True)

'''
Returns the non-numeric attributes.
'''
def getNonNumericAttrs(df):
    return __getNumericHelper(df, False)

def __getNumericHelper(df, findNumeric):
    isNumeric = df.applymap(np.isreal) # np.isreal is a function that takes a value and returns True (the value is real) or False
                                       # applymap applies the given function to the whole data frame
                                       # So this returns a DataFrame of True/False values indicating for each value in the original DataFrame whether it is real (numeric) or not

    isNumeric = isNumeric.all() # all: For each column, returns whether all elements are True
    attrs = isNumeric.loc[isNumeric==findNumeric].index # selects the values in isNumeric that are <findNumeric> (True or False)
    return attrs

# =============================================================================

if __name__ == "__main__":
    main()

