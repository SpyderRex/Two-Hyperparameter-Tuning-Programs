# Two-Hyperparameter-Tuning-Programs
Two different yet similar programs that automate the hyperparameter tuning process in a machine learning model.

In these two programs I created a way to loop possible hyperparameter values through a machine learning model and then print the maximum accuracy that results and the combination(s) of hyperparameter values that produced the accuracy. In one program I nest the model selection and training in a single for loop, and in the other I nest it at the end of multiple for loops, one for each hyperparameter. The result is similar to  sklearn's GridSearchCV module.
