# Predicting Customer Churn in SyriaTel to Enhance Customer Retention

## Introduction
This project we will be working with the SyriaTel Customer churn dataset named as Customer_churn, to solve our business problem.
The Aim of this project is to analyse and determine how the SyrianTel company can reduce customer churn, retain customers leading to inceased profits.

## Business Understanding

#### Business problem
The SyriaTel Company is a telecomunication business that would  like to address the issue of custonwer churnig by creating a prediction model to understand why they are facing a high churn rate and their customers preferring their competitors. The main goal of SyriaTel Company is to reducing churn rate, increse customer retention hence increasing profitability 


## Objectives
1. Identify factors that may lead to customer churn

2. To create a model that will predict customer which are at a high risk of churning

3. The relevant steps that should be taken to retain customers

## Data Understanding

Data Understanding is essential for identifying patterns, ensuring quality and making informed decisions based on accurate analysis.




## Data Cleaning

We do data cleaning to remove any duplicates deal with missing values and any other inconsistencies



## Exploratory Data Analysis

Exploratory Data Analysis helps us identify the correlation between the features in the dataset and the distribution of the variables. This is essential for feature engineering and modelling.



### Univariate Analysis

This involves the distribution of each feature of the dataset in order to understand their characteristics and identify any issues such as outliers.




#### Churn Distribution

The churn column is our target variable and it is categorical. We will ude a pie chart to display its distribution.



![Churn_distribution](Images/Cap_1.PNG)




0 is False
1 is True

The pie chart represents the distribution of customers who left(churned), vs those who stayed.
85.5% represents customers who did not churn 
14.5% represents customers who did churn, that is, they left the company

This imbalance suggests that customer retention is high, but the company should still analyse the reasons for churn to improve retention strategies.

### Distribution of  features

import pandas as pd
import matplotlib

print("Pandas version:", pd.__version__)
print("Matplotlib version:", matplotlib.__version__)

#plotting the the disrtibution of the rest of the features
df.drop(columns='churn').hist(figsize=(10,9), color='blue')
#plt.hist(hist_plot,figsize=(8,9), color='blue')
plt.show()

### Distribution of categorical features

#poltting categorical features
cat_features= df.drop('phone number', axis=1).select_dtypes(include=object).columns
for col in cat_features:
    print(col)
    print(df[col].unique())

#### Subscription plan distributions 

There are three categorical features as shown above; state, international plan, voice mail plan that we will be analysing.

We will create a function that will be used in plotting the distribution of categorical features

#creating the function
def plot_cat_distributions(df, feature):
   plt.figure(figsize=(10,9))
   sns.countplot(x=feature, data=df,color='green',order=df[feature].value_counts().index)
   plt.xticks(rotation=90)
   plt.show()

#### distribution for the state column

plot_cat_distributions(df,'state')

#### Finding

Most of the consumers are from West Virginia, Minnesota, New York, Alabama, Wisconsin. 
while California has the list number of consumers.

#### International plan distribution

df['international plan'].value_counts()

#Plotting international plan
plot_cat_distributions(df,'international plan')

#### Finding

Only 323 customers have subscribed to the are  international plans out of 3333.


### Voice Main Plan Distribution

df['voice mail plan'].value_counts()

plot_cat_distributions(df, 'voice mail plan')

#### Finding

Only 922 customers have subscribed to the voive mail plan services out of 3333.

#### Conclusion
From the above distributions we can conclude that there are more subscribers of voice mail plan compared to the international plan subscribers.

#### Minutes Distribution

# define the column names
column_sum= [ 'total day minutes', 'total eve minutes', 'total night minutes', 'total intl minutes']
# sum for each column
sums= df[column_sum].sum()
plt.figure(figsize=(8,6))
#plot the histogram bars
bars= plt.bar(sums.index, sums, color='purple')
plt.xlabel('columns')
plt.ylabel('Total minutes')
plt.suptitle('Total Minutes Distribution')

#Add percentage labels
for bar in bars:
    height= bar.get_height()
    plt.text(bar.get_x()+ bar.get_width()/2., height+10,f"{height/sum(sums)*100:.1f} %", ha='center', va='bottom')
plt.xticks(rotation=45)
plt.show()

The above bar graph shows total minutes across different call categories.
The total evening minutes have the highest percentage compared to other call categories. This shows that most customers spend more time on calls in the evening. 

### Correlation of Features

We will look at features that have the highest correlation with the target variable(churn)

#numeric columns
numerical_cols= df.select_dtypes(include=['number'])

#the correlation matrix
corr_matrix= numerical_cols.corr()

#plot the heatmap
plt.figure(figsize=(9,8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.suptitle('Features correlation Matrix')
plt.show()

Most features have low correlation with each other close to zero.

There is perfect positive correlations  between:

total eve charge and total eve minutes

total day charge and total day minutes

total night charge and total night minutes

total intl charge and total intl minutes

This is expected because call charges depend on call minutes.

### Customer Service Calls vs Churn

#churn rate percentage for each customer service  call
churn_rate= df.groupby('customer service calls')['churn'].mean()*100

#Plot a bar plot
churn_rate.plot(kind='bar', figsize=(10,9), color='purple')

#title and labels
plt.suptitle('Churn rate by Customer Service Call')
plt.xlabel('Number of customer service calls')
plt.ylabel('Churn Rate in percentage')

plt.show()

#### Finding

As the number of calls increases the rate of churn also increases. However after the sixth call the rate of churn is evident suggesting that more interactions could make customers more unhappy, causing more people to churn. 

#### Preparing  data for machine learning

### Multicollinearity of features

# independent variables to check for multicollinearity
X= df[['total day minutes', 'total eve minutes', 'total night minutes', 'total intl minutes']]

#Calculate VIF for each variable
vif_data = pd.DataFrame()
vif_data['feature'] = X.columns
vif_data['VIF']= [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
vif_data

# VIF is  the measure of multicollineality in a regression model
#

# drop columns with high correlation
to_drop = ['total day charge', 'total eve charge', 'total night charge', 'total intl charge']
df= df.drop(to_drop, axis=1)
df.head()

#looking at the data types
df.dtypes

# the dummy variables

df = pd.get_dummies(df, columns=['state', 'international plan', 'voice mail plan'], drop_first = True)

#drop the phone number column
df.drop('phone number', axis=1, inplace= True)

df.info()

#### Train Test Split 

# split the dataset into target variable(y)and features(X)
y= df['churn']
X= df.drop(columns=['churn'])


#split the data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#### Scaling

# create a standardScaler object
scaler= StandardScaler()

#fit and transform the features
X_scaled= scaler.fit_transform(X)

#Convert scaled features back to a DataFrame
X_scaled= pd.DataFrame(X_scaled, columns=X.columns)
X_scaled.head()



#### To check for model imbalance

#Find the distribution of the target variable
class_distribution = df['churn'].value_counts()
#check if the dataset is imbalanced
if class_distribution[0] / class_distribution[1] > 2 or class_distribution[1]/ class_distribution[0]>2:
    print('Imbalanced dataset')
else:
    print('Balanced dataset')

### Data modelling

#### Logistic regression


###  Dealing with the imbalance

#### Logistic regression with class weights

We can adjust models to handle imbalance.

We will use Class weights to balance. This method assigns a higher weight on the minority class

#logistc regression model
logreg = LogisticRegression(class_weight= 'balanced', random_state=42)
#fit the model
logreg.fit(X_train, y_train)


#generate predictions on the test set
y_pred = logreg.predict(X_test)

# find the performance metrics
#Accuracy
accuracy= accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
# The precision
precision = precision_score(y_test, y_pred)
print('Precision:', precision)
# The recall
recall= recall_score(y_test, y_pred)
print('Recall:', recall)
#the f1 score
F1= f1_score(y_test, y_pred)
print('F1_score:', F1)

# AUc for logistic regression
auc_lr= roc_auc_score(y_test, y_pred)
print('auc_lr:', auc_lr)



####  Confusion Matrix

#generate predictions on the test set
y_pred = logreg.predict(X_test)

#confusion matrix
conf_matrix= confusion_matrix(y_test, y_pred)

#visualization of the matrix
plt.figure(figsize=(9,8))
sns.heatmap(conf_matrix, annot=True, cmap='Reds', fmt='g', cbar=False,
            xticklabels=['Predicted Negative', 'Predicted Positive'],
            yticklabels=['Actual Negative', 'Actual Positive'])
plt.xlabel('Predicted label')
plt.ylabel('True label')
#plt.title( 'Confusion Matrix')
plt.show()

A higher number of true positives and true negatives indicates better performance , while the false positives or negatives highlight ares for improvement.

### Random Forest

 Since we are using Random Forest, we will balance the dataset this time by class weights

#### Balancing the dataset

# random forest with class weights
rf = RandomForestClassifier(class_weight='balanced', random_state=42)
# Train the Random Forest model on training data
rf.fit(X_train, y_train)


#generate predictions on the test data
y_pred_rf= rf.predict(X_test)

# find accuracy
accuracy_rf= accuracy_score(y_test, y_pred_rf)

#find the precision
precision_rf= precision_score(y_test, y_pred_rf)

#find the recall
recall_rf= recall_score(y_test, y_pred_rf)

# find the f1_score
f1_rf= f1_score(y_test, y_pred_rf)

# find the roc and auc score
roc_auc_rf= roc_auc_score(y_test, y_pred_rf)

#print the Random Forest metrics

print('Random Forest Metrics:')
print( 'Accuracy:', accuracy_rf)
print('Precision:', precision_rf)
print('Recall:', recall_rf)
print('F1 score:', f1_rf)
print('ROC AUC Score:', roc_auc_rf)

#### points to note

The Random Forest model has an accuracy score of 92.2%, meaning its is able to correctly classify instances.

The model has achieved a precision of 96.2% showing a high rate of correctly prediting churn instance out of all the predicted churn instances.

The Recall is 50.5% which suggests that the model captures about half of the actual churn instances.

The F1 score is 66.2% which represents the balance between precision and recall, indicating moderate performance.

The ROC AUC Score is 75.1% suggesting that the model performs reasonably well in disguishing between churn and non-churn instances.

### Confusion Matrix

#Random Forest Confusion Matrix
con_matrix_rf= confusion_matrix(y_test, y_pred_rf)

#Visualize the matrix
plt.figure(figsize=(9,6))
sns.heatmap(con_matrix_rf, annot= True, cmap='Reds', fmt='g', cbar=False,
           xticklabels=['Predicted Negative', 'Predicted positive'],
           yticklabels=['Actual Negative', 'Actual Positive'])
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.suptitle('Confusion Matrix - Random Forest Model')
plt.show()


#### points to note

-There is a high number of true negatives and true posistives, indicating good performance in predicting both non-churn and churn instances.

-There is a small number of false positives and false negatives, suggesting that the model has relatively low misclassification rates. 

### Hyperparameter tuning

#define the parameter grid
param_grid= {
    'n_estimators':[50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split':[2, 5, 10],
    'min_samples_leaf':[1, 2, 4,],
    'max_features':['auto', 'sqrt', 'log2']}
#create the Random forest classifier
rf_model = RandomForestClassifier(random_state=42)
#grid search with five-fold cross-validation
grid_search= GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring= 'accuracy', n_jobs=-1)
#Fit the grd search to the data
grid_search.fit(X_train, y_train)
#Find the AUC score for the tuned Random Forest Classifier
auc_rf_tuned = roc_auc_score(y_test, y_pred_rf)
#get the best parameters and the best score
best_params= grid_search.best_params_
best_score= grid_search. best_score_

print("Best parameters:", best_params)
print("Best Accuracy Score:", best_score)
print("AUC:", auc_rf_tuned)

        

### Decision Tree Model

#instantiate the DecisionTreeClassifier
dt_model= DecisionTreeClassifier(class_weight= 'balanced', random_state= 42)
#fit the model to the training data
dt_model.fit(X_train, y_train)

#generate Predictions on the test set
y_pred_dt= dt_model.predict(X_test)

#calculate the performance metrixs

accuracy_dt= accuracy_score(y_test, y_pred_dt)
precision_dt= precision_score(y_test, y_pred_dt)
recall_dt= recall_score(y_test, y_pred_dt)
f1_dt= f1_score(y_test, y_pred_dt)

#AUC for decision Tree
auc_dt= roc_auc_score(y_test, y_pred_dt)

# print performance metrics
print('Accuracy:', accuracy_dt)
print('Precision:', precision_dt)
print('Recall:', recall_dt)
print('F1 Score:', f1_dt)
print('auc_dt', auc_dt)

#### Interpretation

The accuracy achieved 93.3%  is higher than the accuracy of random forest and Logistic regression.

The precision achieved is 78% indicating a lower rate of false positives, compared to logistic regression which had about 51.5%

The recall is 77.2% suggesting that the model captures about 77.2% of the churn instances.

F1 score is 77.6% which show the balance between the recall and the precision.

### Confusion Matrix

# Predictions on the test set
y_pred_dt = dt_model.predict(X_test)

#Building the confusion matrix
conf_matrix_dt= confusion_matrix(y_test, y_pred_dt)

#Visualise the matrix
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix_dt, annot=True,cmap='Reds', fmt='g', cbar=False,
           xticklabels=['Predicted Negative', 'Predicted Positive'],
           yticklabels=['Actual Negative', 'Actual Positive'])
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.suptitle('The Decision Tree Matrix')
plt.show()

The model above correctly identified 78 instances as positive 

It correctly classified 544 instatnces as negative(non_churn) outnof the actual negative instances.

There were 22 instances incorrectly classified as positive (False positives).

There are 23 instances incorrectly classified as negative(False negatives)

### AUC values of logistic regression, Random Forest and Decision Tree Model

#fit the random forest model to the training data
rf_model.fit(X_train, y_train)
#Predicted probabilities for logistic regression
y_prob_lr= logreg.predict_proba(X_test)[:,1]
#predicted probabilities for Random Forest
y_prob_rf= rf_model.predict_proba(X_test)[:,1]
#predicted probabilities for Decision Tree
y_prob_dt= dt_model.predict_proba(X_test)[:,1]

# ROC curve for Logistic Regression
fpr_lr, tpr_lr,_=roc_curve(y_test, y_prob_lr)
# ROC curve for Random Forest
fpr_rf, tpr_rf,_ = roc_curve(y_test, y_prob_rf)
# ROC  Curve for Decision Tree
fpr_dt, tpr_dt,_ = roc_curve(y_test, y_prob_dt)


# AUC ROC scores
auc_lr = roc_auc_score(y_test, y_prob_lr)
auc_rf = roc_auc_score(y_test, y_prob_rf)
auc_dt = roc_auc_score(y_test, y_prob_dt)

#plot ROC curves
plt.figure(figsize=(10,9))
plt.plot(fpr_lr, tpr_lr, label= f'Logistic Regression (AUC = {auc_lr:.2f})')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest(AUC = {auc_rf:.2f})')
plt.plot(fpr_dt, tpr_dt, label=f' Decision Tree(AUC = {auc_dt:.2f})')
         
#Plotting ROC curve for random guessing
plt.plot([0, 1], [0, 1], linestyle='--', color= 'green')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.suptitle('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()


The Random Forest curve displays the best.

### Model Evaluation

#### Logistic Regression

Accuracy: 0.75,
Precision: 0.34,
Recall: 0.69,
F1_score: 0.46,
auc_lr: 0.73.

Summary:The model has a decent overall accuracy of 75%, but struggles  with low precision, leading to relative;y low F1 score and precision


#### Random Forest


Accuracy:0.92, precison:0.96, Recall:0.50, F1 score:0.66, 

The model performs well with high accuracy and precision, but has a moderate recall, indicating missed true positives.


#### Decision Tree

Accuracy: 0.93, Precision:0.78, Recall:0.77, F1 score: 0.78

The model achieves high accuracy with a good balance between precision and recall
and strong AUC performance.
                


#### Summary on Model Evaluation

The decision tree is the best model since it gives the best accuracy and strong AUC

### Conclusion

-Desision Tree has emerged as our best model.

-West virginia has the highesrt number of customers while Califonia has the lowest number of customers.

-There are more total evening calls compared to the rest of the calls, meaning that the syria tel customers prefer evening calls.

-There are more subcriptions to voice mail plan than international plan.

-As the number of customer service call increases the number of churn increases. This shows that the customers experience negative customer service, where their issues are not solved leading to increase in customer churn.


### Recommendatations

-The best model to be used to predict customer churn is The Decision Tree Model, since, it has a good balance between precision and recall, identifying the positive class instances and minimizing the false negatives and false positivesand has a 92% accuracy hence can predict churn.

-Most customers neither have an international plan nor a voice mail plan. The SyriaTel company should consider promotion services to their customers by displaying the benefits hence attracting more.

-The SyriaTel Company should identify influential predictors and usage patterns identified by the model to develop retention strategies.

-The company should put in place systems and the qualified personnel for customer serve, in order to asolve the customers needs fast and efficiency to ensure satisfaction hence customer retention.

### Future Steps

The Decision Tree Model has good accuracy and performance, despite this the company should continue monitoring and evaluating its performance on new data. This is because customer behaviours and preferences shift over time hence ensure the model remains effective and up-to-date.