# Binary Classifier Class

Binary classification model which wraps python scikit-learn's **Regularized Logistic Regression**
with series of specific functionalities. 

All input features `X` are assumed to be given as pandas DataFrames 
with a mix of numeric and categorical data. The output `y` is assumed to be in form of a numpy array of 0-s and 1-s.
```python
X = pd.DataFrame({'v1': ['c', 'b', 'a'], 'v2': [2, 6, 4]}) 
y = np.array([0, 0, 1])
```
The package is organized in following folders:
 - `src`: Main class codes and utilities 
 - `test`: Test cases for unittest 
 - `docs`: Documentation and questions
 - `data`: Test data and exploratory data analysis
 
 There are two more classes developed beside the main classifier classes as below 
 which are hosted under *robo_prep* file.
 - `RoboImputer`: Filling up the missing values
 - `RoboFeaturizer`: generalized pre-processing of input data for classifier class
 
## Imputer
The class *RoboImputer* performs filling up the missing values with the laws the simple imputing rules below.
This class is used in pro-processing of data thorough the *RoboFeaturizer* class.   
 - Column dtype    :   Imputing Value
 - object/category :   most frequent value in the column
 - float           :   mean of column values
 - other dtype     :   median of column
 
 Initialization syntax:
```python
i = RoboImputer()
```  

## Featurizer
All the generalized pre-processing of input data for classifier class are taken care of using 
the *RoboFeaturizer* class which also gets helped in imputing missing values by the auxiliary 
`RoboImputer` class above.

Initialization syntax: 
```python
f = RoboFeaturizer(
             max_unique_for_discrete=10,
             max_missing_to_keep=0.80,
             add_missing_flag=False,
             encode_categorical=True,
             max_category_for_ohe=10,
             scaler = StandardScaler()
             )
```
where 
 - `max_unique_for_discrete` [int]: Encode columns of up to this threshold of unique numeric values, using one-hot-encoding.  
 - `max_missing_to_keep` [0=<float<=1]: Drop columns if fraction (%) of missing values are higher than this threshold
 - `add_missing_flag` [Bool]: For columns with missing values smaller than the threshold, add a binary column to mark missing data points. 
 - `encode_categorical` [Bool]: Encode categorical (non-numerical) columns
 - `max_category_for_ohe` [int]: Encode categorical columns with number of categories up to this threshold, use one-hot-encoding and for the rest use alternate. Only effective if `encode_categorical=True` 
 -  `scaler` [None or sklearn scaler]:  No scaling of input if None, scale input using 'fit_transform' method



## Classifier based on Regularize Logistic Regression
The class follows standard scikit-learn template in initialization and method definition.
Custom-defined methods for this class are listed below:

Initialization syntax: 
```python
f = RoboLogistic(
             max_unique_for_discrete=10,
             max_missing_to_keep=0.80,
             add_missing_flag=False,
             encode_categorical=True,
             max_category_for_ohe=10,
             scaler = StandardScaler(),
             ...)
```
where the initialization parameters are combination of `RoboFeaturizer` class discussed above and 
scikit-learn's `LogisticRegression` class.
 
##### fit: 
Fit on training data.
```python
self.fit(X, y)
```
##### predict:
Predict class labels on new data.
```python
self.predict(X)
```
##### predict_proba:
Predict the probability of each label.
```python
self.predict_proba(X)
```

##### evaluate:
Find the value of the F1-score, LogLoss metric.
```python
self.evaluate(X, y)
```
##### tune_parameters:
Run K-fold cross validation to choose the best parameters given k as number of folds and 
dictionary of hyper-parameters for search.
```python
self.tune_parameters(X, y, k=5, hypeparam_grid=None)
```

## Unit Test
Exploratory data analysis of test dataset was performed and provided for reference in  
[a notebook](../data/EDA.ipynb).

For the purpose of unit test, a simple class of `TestData` was developed to load either the bad credit dataset or 
the breast cancer dataset from scikit-learn. The class simply loads the data and helps with generating *X* and *y*
Unit test was developed with a focus on each of the three classes defined and discussed above. 

#### TestData Class
```python
X, y = TestData().make_X_y()
```

#### Test Cases
For each of the classes:
- Check if proper validation and error is provided if a class instance is not fit and called for prediction or transform. 
- Check if input is in the correct shape during calling transform or prediction.
- Check class-specific cases to ensure class is behaving according to the design.

There is room for more test cases and integration with this package. 
Coommand below can be used from command line to start testing: 
```cmd
$ python -m unittest discover
```

##### Tests on Imputer Class
 - `test_bad_input_shape`
 - `test_unfit_instance`
 - `test_filevalue1`
 - `test_filevalue2`

##### Tests on Featurizer Class
 - `test_bad_input_shape`
 - `test_unfit_instance`
 - `test_missing_value`
 - `test_new_category`

##### Tests on Logistic Class
 - `test_bad_input_shape`
 - `test_unfit_instance`
 - `test_result_formats`
 - `tesT_reproduce`