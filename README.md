# Machine-Learning
A small collection of codes I've written for Machine Learning Course Instructed by Dr. Hamed Malek.

## News Classification
Introduction:

In this task, we have a large (2 GB) dataset where one column contains the text of news articles, and another column indicates the category or topic of the news. The goal of this project is to develop a neural network model to classify news articles into their respective categories. The dataset has been preprocessed with Tokenization and Feature Extraction.
### Initial Processing:
The first step involves improving and refining the category labels. A review of the titles shows that some titles are synonymous or belong to the same broader category. These titles are merged to improve the accuracy of the classification. The result is a set of main categories, and any null values in the dataset are identified and removed.
before merge :
![image](https://github.com/user-attachments/assets/39efe331-69d0-4cf5-af05-06aecf6c796c)

after merge :

![image](https://github.com/user-attachments/assets/d597e3d1-edb8-4f37-898a-e5cffec5aed8)

### Train, Test, and Split:
To optimize the model and reduce overfitting, the dataset is split into training and testing sets. The goal is to ensure that the training data contains an equal number of samples from each category, making the data balanced. The dataset is then split into training and testing sets, with the training set representing 80% of the data and the testing set representing 20%.

![image](https://github.com/user-attachments/assets/9fb37e1f-a67a-490a-a1cf-b4db5e798a67)

### Tokenization and Feature Extraction:
Tokenization is performed to break down the text into words or tokens, and the TF-IDF (Term Frequency-Inverse Document Frequency) method is used for feature extraction. This method highlights the importance of a word within a document relative to a collection of documents. Various parameters like ngram_range and max_features are tuned to control the tokenization process and the number of features extracted. It is important to mention that using TF-IDF after seperating train and test results in more negatives comparing to real - life use.
### Artificial Neural Network (ANN):
The neural network architecture is designed for the task. Recurrent Neural Networks (RNNs) are typically used for natural language processing tasks, but the complexity of such networks can lead to overfitting. To prevent this, the network architecture includes dropout layers, which deactivate a fraction of the neurons during training to avoid overfitting. The final output layer uses a softmax activation function to classify the input into one of the predefined categories. We use adam as optimizer and categorical cross entropy for loss.

![image](https://github.com/user-attachments/assets/02bb385f-9c73-484b-8eec-65f709a941ef)


### Model Evaluation:

The model is evaluated using the following methods:

    Cross-Validation: A common method in machine learning to assess how the model generalizes to an independent dataset.
    Confusion Matrix: A tool to visualize the performance of the classification model by showing the correct and incorrect predictions.
    K-Folds Cross-Validation: The dataset is split into k parts, where each part is used as a test set while the rest are used as the training set. This method provides a more robust evaluation of the model's performance.

### User Interface Implementation:

A simple user interface is created using the Gradio library, which allows for easy integration of the model into an interactive environment, such as Google Colab, where users can input new text data and receive classification predictions from the model.

## DBSCAN
We are tasked to implement the DBSCAN algorithm in order to cluster two provided datasets. First, we create a scatter plot of these datasets to visualize their dispersion. Then,we apply the DBSCAN algorithm to perform clustering on the two datasets. Finally, the identified clusters are displayed through highlighting them with different colors to distinguish between them.
Further details and results are provided within the notebook.

## SVM
A stroke, also known as a cerebrovascular accident, occurs when a part of the brain is deprived of its blood supply, causing the part of the body controlled by the affected brain cells to stop functioning. This loss of blood supply can be due to a lack of blood flow or bleeding in the brain tissue. A stroke is a medical emergency because it can lead to death or permanent disability. There are treatments available for this type of stroke, but they must be initiated within a few hours after the symptoms appear.
Provided with a dataset named strokes.csv, which includes information about individuals and their stroke history, our code implemented these steps:

a. Perform preprocessing operations according to the problem's objective.
b. Split the data into training and test sets with an appropriate ratio.
c. Train a suitable model using the SVM algorithm from the sklearn library.

Further details and results are provided within the notebook.
## Gradient Descent
In the health insurance industry, insurance companies often face challenges in accurately determining the premium for each policyholder. Mistakes in assessing patients' health risks can lead to significant financial losses. Therefore, accurately determining health insurance premiums is crucial for maintaining the financial stability of insurance companies and providing fair services to policyholders.

In this project, we used a dataset containing information about health insurance policyholders, including age, number of children, smoking habits, residential area, gender, body mass index (BMI), and the medical expenses provided by the insurance. This dataset serves as a valuable resource for developing predictive models that can help health insurance companies assess risks and determine more accurate premiums. This project directly impacts the operational and business strategy of health insurance companies, ultimately benefiting both the companies and their policyholders.

The objectives of this project are to develop machine learning models that can help health insurance companies with the following:

    Accurate Premium Determination: We used policyholder data to more accurately calculate premiums based on the health risks each insurer faces. As a result, insurance companies can minimize financial losses caused by incorrect premiums.
    Health Risk Assessment: We identified risk factors that affect individual medical expenses, such as age, BMI, number of children, and smoking habits. This can help insurance companies assess and manage risks more effectively.

Provided with a dataset named insurance.csv, which contains information about policyholders collected by one of the health insurance companies, our code implemented these steps:

a. Perform preprocessing operations according to the problem's objective.
b. Split the data into training and test sets with an appropriate ratio (report the reason for choosing the specific training-test ratio).
c. Implement a linear regression model based on the concepts taught in class and report the model's accuracy on the test data.
d. Extend your previous model to a polynomial regression and report the accuracy.

Further details and results are provided within the notebook.
## KNN
Implementation of the K-Nearest Neighbors (KNN) algorithm from scratch on Desicion tree dataset. The accuracy of model on the test data is reported using the functions from the Scikit-learn library. The ROC curve for Trained KNN model is plotted.The ROC (Receiver Operating Characteristic) curve is a graphical plot that illustrates the diagnostic ability of a binary classifier system as its discrimination threshold is varied.

Key concepts and uses of the ROC curve include:

    Classification Accuracy:
        The ROC curve shows how well the model distinguishes between the positive and negative classes across different threshold levels.
        Accuracy is one of the important metrics used in this context.

    AUC (Area Under the Curve):
        AUC represents the area under the ROC curve and ranges between 0 and 1.
        The closer the AUC is to 1, the better the model's performance.

This curve helps evaluate the trade-off between true positive rates and false positive rates, providing a clear picture of the model's classification power. The closer the curve is to the top-left corner, the better the model's performance, indicated by a higher AUC value.
## Desicion Tree
In this project, we aim to implement a decision tree from scratch for the purpose of multi-class classification in the following steps :
a. Implement a decision tree model.
b. Train this model on the given dataset and test the trained model with different hyperparameters.
c. Using existing libraries, train Random Forest and Gradient Boosting models on this dataset. Then, train our models on 25%, 50%, 75%, and 100% of the data, and plot the Learning Curve separately for each. 
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

Criterion: This is the function used to measure the quality of a split. Two common options are:

    Gini: Measures the impurity of a node by the probability of incorrectly classifying a randomly chosen element if it were randomly labeled according to the distribution of labels in the node.
    Entropy (Information Gain): Measures the information gained from a split, calculated as the reduction in entropy (or uncertainty) after the split.

The Gini index is more computationally efficient, while the entropy criterion (also known as log loss in some contexts) is more theoretically sound but may be slower to compute.
The criterion parameter in Scikit-learn allows you to choose between "gini" and "entropy" (log loss).
Our model is also evaluated using learning curves.
A learning curve is an essential tool to assess the performance of a model. It helps in understanding how well the model is learning with increasing amounts of training data and can highlight issues such as:

    Underfitting: Occurs when both the training score and cross-validation score are low.
    Overfitting: Happens when the training score is high, but the cross-validation score is significantly lower.

we plotted learning curves using 25%, 50%, 75%, and 100% of the training data to evaluate how the model's performance scales with more data. 
Further details and results are provided within the notebook.
