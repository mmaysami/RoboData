
## Further Evaluation
##### Question 1 
Imagine the credit risk dataset with binary target is defined as bad credit or 
*whether the loan defaulted or payments were missed*. Two models are developed: 
a logistic regression model with an F1-score of 0.60 and a neural network with an 
F1-score of 0.63. Which model would you recommend for the bank and why?

**A.** Given the bank strategy is to minimize risk and avoid providing load to 
customers with bad credit, they would want to minimize the value of false negatives,
which translates to maximizing recall over precision. It would be recommended to identify 
the recall value for both classifiers to provide a better recommendation. However, if that 
is not an option and we are limited to F1-score which combines the two to provide a 
unified score on the trade-off of the two and generally speaking higher F1-score means 
better accuracy of prediction, the neural network with F1-score=0.63 would be recommended.

##### Question 2
Assume, that several models are created for a dataset such as random forest, 
linear regression, and neural network regressor. What would be a good suggestion 
to get feature importance for this dataset with a good python knowledge?  

**A.** There are multiple methods in feature_Selection module of scikit-learn to rank or select top features.
*Recursive Feature Estimation (RFE)* is one of more common examples for feature selection which 
iteratively takes the estimator, fitting it to the data, and removing the features with 
the lowest weights (coefficients). 
This process requires estimator to have a `coef_` or  `feature_importances_` attribute.

 - Linear Regression model provides `coef_` attribute after fit that can be used to achieve the goal of ranking input features.
 - Random Forest model provides `feature_importances_` attribute after fitting which can be used to 
 rank the features and pick the top ones.
 - Neural Network model also has `coefs_` that can provide some idea of feature importance.
  