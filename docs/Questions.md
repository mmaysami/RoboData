
## Further Evaluation
##### Question 1 
Imagine the credit risk dataset with binary target is defined as bad credit or 
*whether the loan defaulted or payments were missed*. Two models are developed: 
a logistic regression model with an F1-score of 0.60 and a neural network with an 
F1-score of 0.63. Which model would you recommend for the bank and why?

A. Given the bank strategy is to minimize risk and avoid providing load to 
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

A.