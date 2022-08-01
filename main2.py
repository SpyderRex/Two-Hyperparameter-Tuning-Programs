#import necessary libraries 
import itertools
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

#load any dataset
digits = load_digits()
df = pd.DataFrame(digits.data, columns=digits.feature_names)
df["target"] = digits.target

#prepare the data for training
X = df.drop(["target"], axis="columns")
y = df.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#create lists of chosen values for hyperparameters to be tuned in intended model 
C = [1, 5, 10, 20, 50]
gamma = ["scale", "auto", 0.001, 0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
kernel = ["linear", "poly", "rbf", "sigmoid"]

#create a list of all possible combinations of hyperparameter values
a = [C, gamma, kernel]

combined_hyperparameter_list = list(itertools.product(*a))

#train the chosen model on every combination of hyperparameters and create a list of model accuracies by nesting the model selection at the end of multiple for loops, one for each hyperparameter that will be tuned
accuracy_list = []
for c in C:
    for g in gamma:
        for k in kernel:
            model = SVC(C=c, gamma=g, kernel=k)
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)
            accuracy_list.append(accuracy)

#create a function to merge the list of combined hyperparameter values with the list of accuracies, and then call the function
def merge(list1, list2):
    merged_list = [(list1[i], list2[i]) for i in range(0, len(list1))] 
    return merged_list 
    
newList = merge(combined_hyperparameter_list, accuracy_list)

#find the maximum accuracy and print it, along with the combination of hyperparameter values that produced this accuracy (there are normally more than one combination that will produce the same maximum accuracy)
mx = max(accuracy_list)
print(mx)
print()
for i in newList:
    if i[1]  == mx:
        print(i[0])