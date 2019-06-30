class RoboFeaturizer(object):
    pass# Binary Classifier Class

Binary classification model which wraps python scikit-learn's **Regularized Logistic Regression**
with series of specific functionalities.

All input features `X` are assumed to be given as pandas DataFrames 
with a mix of numeric and categorical data. The output `y` is assumed to be in form of a numpy array of 0-s and 1-s.
```python
X = pd.DataFrame({'v1': ['c', 'b', 'a'], 'v2': [2, 6, 4]}) 
y = np.array([0, 0, 1])
```

## Imputer
The *RoboFeaturizer* class usese the *RoboImputer* class to fill up the missing values which follows the simple imputing rules below:   
 - Column dtype    :   Imputing Value
 - object/category :   most frequent value in the column
 - float           :   mean of column values
 - other dtype     :   median of column
 
 Initialization syntax:
```python
i = RoboImputer()
```  

## Featurizer
All the pre-processing required is performed thorough an auxilliary class `RoboFeaturizer` and `RoboImputer`.

Initialization syntax: 
```python
f = RoboFeaturizer(
             max_unique_for_discrete=10,
             max_missing_to_keep=0.80,
             add_missing_flag=False,
             encode_categorical=False,
             max_category_for_ohe=10,
             )
```
where 
 - `max_unique_for_discrete` [int]: Encode columns of up to this threshold of unique numeric values, using one-hot-encoding.  
 - `max_missing_to_keep` [0=<float<=1]: Drop columns if fraction (%) of missing values are higher than this threshold
 - `add_missing_flag` [Bool]: For columns with missing values smaller than the threshold, add a binary column to mark missing data points. 
 - `encode_categorical` [Bool]: Encode categorical (non-numerical) columns
 - `max_category_for_ohe` [int]: Encode categorical columns with number of categories up to this threshold, use one-hot-encoding and for the rest use alternate. Only effective if `encode_categorical=True` 
 <!-- - `sparse` [Bool]: -->


## Classifier based on Regularize Logistic Regression
The class follows standard scikit-learn template in initialization and method definition.
Custom-defined methods for this class are listed below:


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
Run K-fold cross validation to choose the best parameters.
```python
self.tune_parameters(X, y)
```

## Test Data
Exploratory data analysis of test dataset has been performed and provided in  
[a notebook](data/EDA.ipynb).
