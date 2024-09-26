# Machine-Learning
A Collection of codes I've written for Machine Learning Course Instructed by Dr. Hamed Malek.

## DBSCAN
We are tasked to implement the DBSCAN algorithm in order to cluster two provided datasets. First, we create a scatter plot of these datasets to visualize their dispersion as shown below :

<img src ="https://github.com/user-attachments/assets/e3d778e9-e9d5-4f80-98ab-d0dfc7d23b6d" width ="400"> 


<img src="https://github.com/user-attachments/assets/93c92885-6c6e-4038-a5de-f910218c0195" width ="400">


Then,we apply the DBSCAN algorithm to perform clustering on the two datasets. Finally, the identified clusters are displayed through highlighting them with different colors to distinguish between them.
Further details and results are provided within the notebook.


<img src ="https://github.com/user-attachments/assets/ca1e226e-688f-4e99-87cc-67e5becfacd8" width ="400">


<img src="https://github.com/user-attachments/assets/b966483f-8d44-4900-b9c8-a7f62c4a33c0" width ="400">


## SVM
A stroke, also known as a cerebrovascular accident, occurs when a part of the brain is deprived of its blood supply, causing the part of the body controlled by the affected brain cells to stop functioning. This loss of blood supply can be due to a lack of blood flow or bleeding in the brain tissue. A stroke is a medical emergency because it can lead to death or permanent disability. There are treatments available for this type of stroke, but they must be initiated within a few hours after the symptoms appear.

<img src ="https://github.com/user-attachments/assets/7c637b29-1039-4aea-a231-cc3e3e569c3a" width ="400">

Provided with a dataset named strokes.csv, which includes information about individuals and their stroke history, our code implemented these steps:

a. Performing preprocessing operations according to the problem's objective.
b. Splited the data into training and test sets with an appropriate ratio.
c. Trained a suitable model using the SVM algorithm from the sklearn library.

Further details and results are provided within the notebook.

<img src ="https://github.com/user-attachments/assets/f0ebb766-843e-49f5-a02f-2c367dbc69e5" width ="400"> 


<img src="https://github.com/user-attachments/assets/9c54edb4-658f-41c5-87fd-4fd0afc0077a" width ="400">


## Gradient Descent
In the health insurance industry, insurance companies often face challenges in accurately determining the premium for each policyholder. Mistakes in assessing patients' health risks can lead to significant financial losses. Therefore, accurately determining health insurance premiums is crucial for maintaining the financial stability of insurance companies and providing fair services to policyholders.

In this project, we used a dataset containing information about health insurance policyholders, including age, number of children, smoking habits, residential area, gender, body mass index (BMI), and the medical expenses provided by the insurance. This dataset serves as a valuable resource for developing predictive models that can help health insurance companies assess risks and determine more accurate premiums. This project directly impacts the operational and business strategy of health insurance companies, ultimately benefiting both the companies and their policyholders.
### Dataset
Before :

<img src ="https://github.com/user-attachments/assets/03fc2752-cce3-4df4-a5de-e03e525c3217" width ="400">

After :

<img src ="https://github.com/user-attachments/assets/55487099-04b8-4a15-a2ad-fae2fcf09694" width ="400">

The objectives of this project are to develop machine learning models that can help health insurance companies with the following:

    Accurate Premium Determination: We used policyholder data to more accurately calculate premiums based on the health risks each insurer faces. As a result, insurance companies can minimize financial losses caused by incorrect premiums.
    Health Risk Assessment: We identified risk factors that affect individual medical expenses, such as age, BMI, number of children, and smoking habits. This can help insurance companies assess and manage risks more effectively.

Provided with a dataset named insurance.csv, which contains information about policyholders collected by one of the health insurance companies, our code implemented these steps:

a. Performed preprocessing operations according to the problem's objective.
b. Splited the data into training and test sets with an appropriate ratio.
c. Implemented a linear regression model.
d. Extended our previous model to a polynomial regression and reported the accuracy.

### Linear Regression model result

<img src ="https://github.com/user-attachments/assets/442d2bfa-1389-4cf6-ae61-781aca33d77a" width ="400">

### Different Split size results

<img src ="https://github.com/user-attachments/assets/6d22b962-7084-4417-9c94-c992d69f3479" width ="400">

### Polynominal Implementation results

<img src ="https://github.com/user-attachments/assets/7390e202-0281-4c77-8ab2-c747b6bd8004" width ="400">

Further details and results are provided within the notebook.
## KNN
Implementation of the K-Nearest Neighbors (KNN) algorithm from scratch on Desicion tree dataset. The accuracy of model on the test data is reported using the functions from the Scikit-learn library. The ROC curve for Trained KNN model is plotted.The ROC (Receiver Operating Characteristic) curve is a graphical plot that illustrates the diagnostic ability of a binary classifier system as its discrimination threshold is varied.
### Dataset
before :

<img src ="https://github.com/user-attachments/assets/b6e3dff9-6b21-4b67-b1d5-614117a53964" width ="400">

after :

<img src ="https://github.com/user-attachments/assets/e6b6617f-f333-4020-a4fa-9bc93a9a8a2c" width ="400">

Key concepts and uses of the ROC curve include:

    Classification Accuracy:
        The ROC curve shows how well the model distinguishes between the positive and negative classes across different threshold levels.
        Accuracy is one of the important metrics used in this context.

    AUC (Area Under the Curve):
        AUC represents the area under the ROC curve and ranges between 0 and 1.
        The closer the AUC is to 1, the better the model's performance.

This curve helps evaluate the trade-off between true positive rates and false positive rates, providing a clear picture of the model's classification power. The closer the curve is to the top-left corner, the better the model's performance, indicated by a higher AUC value.

<img src ="https://github.com/user-attachments/assets/fc0ee073-68e4-478f-8025-ee96f59f41d4" width ="400">


## Desicion Tree
In this project, I implemented a decision tree from scratch for the purpose of multi-class classification in the following steps :
a. Implemented a decision tree model.
b. Trained this model on the given dataset and test the trained model with different hyperparameters.
c. Using existing libraries, trained Random Forest and Gradient Boosting models on this dataset. Then, trained our models on 25%, 50%, 75%, and 100% of the data, and ploted the Learning Curve separately for each. 
The dataset provided includes the following features:

    Disease: The name of the disease or medical condition
    Fever: Indicates whether the patient has a fever (Yes/No)
    Cough: Indicates whether the patient has a cough (Yes/No)
    Fatigue: Indicates whether the patient experiences fatigue (Yes/No)
    Difficulty Breathing: Indicates whether the patient has difficulty breathing (Yes/No)
    Age: The age of the patient in years
    Gender: The gender of the patient (Male/Female)
    Blood Pressure: The blood pressure level of the patient (Normal/High)
    Cholesterol Level: The cholesterol level of the patient (Normal/High)
    Outcome Variable: The outcome variable indicating the result of the diagnosis or assessment for the specific disease (Positive/Negative)

The correctness of the diagnosis of the disease (Disease) is determined in this dataset based on the medical symptoms (other available features) in the column "Outcome Variable." It is important to note that given some features with categorical values (e.g., Male/Female), necessary preprocessing should be performed on the dataset first.
### Hyperparameters for Decision Tree Implementation:
Max Depth: This parameter controls the maximum depth of the tree. For example:

    Depths of 1, 2, and 3 typically result in underfitting, where the model is too simple to capture the data's complexity.
    Depths of 8, 9, and 10 may lead to overfitting, where the model becomes too complex and fits the training data too closely.


<img src ="https://github.com/user-attachments/assets/23252d99-9ae8-4944-b3e1-7093b57f3e8c" width ="400">



Criterion: This is the function used to measure the quality of a split. Two common options are:

    Gini: Measures the impurity of a node by the probability of incorrectly classifying a randomly chosen element if it were randomly labeled according to the distribution of labels in the node.
    Entropy (Information Gain): Measures the information gained from a split, calculated as the reduction in entropy (or uncertainty) after the split.


<img src ="https://github.com/user-attachments/assets/25b41cd9-6bdd-4a02-bc9f-59903d58750f" width ="400">


The Gini index is more computationally efficient, while the entropy criterion (also known as log loss in some contexts) is more theoretically sound but may be slower to compute.
The criterion parameter in Scikit-learn allows you to choose between "gini" and "entropy" (log loss).
Our model is also evaluated using learning curves.
A learning curve is an essential tool to assess the performance of a model. It helps in understanding how well the model is learning with increasing amounts of training data and can highlight issues such as:

    Underfitting: Occurs when both the training score and cross-validation score are low.
    Overfitting: Happens when the training score is high, but the cross-validation score is significantly lower.

we plotted learning curves using 25%, 50%, 75%, and 100% of the training data to evaluate how the model's performance scales with more data. 
Further details and results are provided within the notebook.

<img src ="https://github.com/user-attachments/assets/6e0dc066-5c88-437b-8141-4b2991b4285b" width ="400">


<img src ="https://github.com/user-attachments/assets/db1584a2-cb16-421e-ad53-35e17e053278" width ="400">


<img src ="https://github.com/user-attachments/assets/5efdaad3-9f20-4d78-8e0c-eda896b29887" width ="400">


