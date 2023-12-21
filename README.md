# 🥉 Hackathon of the Graduate School of Business and VK   
Predicting patient recovery using ML algorithms, image recognition for sanctuary by computer vision models, recommendation system for advertising banners 

## 🦸‍♂️ Team
We are the bronze medalists of this competition. 
Get to know us:
- [Solomon](https://github.com/veidlink)
- [Roman](https://github.com/rtccreator)
- [Katerina](https://github.com/dekatrine)
- [Nikita](https://github.com/AnalyseOptimize)

## 🎯 Task 1 | Predicting patient recovery using ML algorithms


The task for VK Predict, a business unit of the company VK, involves developing a predictive model for a pharmaceutical company. This model should predict patient recovery based on test results, specifically for those treated with a new, expensive medication designed to replace a less effective, cheaper one. The data consists of anonymized patient features correlating to test results, with a training dataset (train.csv) including a target feature indicating whether the illness was cured by the medication (values 0 or 1), and a test dataset (test.csv) for predictions. Solution was evaluated using the F1-score metric.

### ⚙️ Tech stack 
- **Scikit-learn** 
- **Catboost** as a baseline approach
- **Perceptron** as the final solution
- **Matplotlib, Seaborn** for EDA
- **Shap** for feature selection and visualisation
- **Optuna, GridSearchCV** for hyperparameter tuning
- **Pytorch** for writing custom FCNN

### Feature selection and some experiments 


We deleted 1557 outliers from the data with IQR (interquartile range). We also dropped feature number 5 because it was highly linearly correlated (unlike others) with the target variable in the training dataset, while this was not observed in the test dataset. The model was overfitting on the 5th feature, which was causing a decrease in performance, and we corrected this.

|   | target |
|---|--------|
| 5 | 0.85876|
| target | 1.00000|

After obtaining the initial result with the CatBoost model tuned using the Optuna library, we visualized the feature importances to test several hypotheses. 

We decided to add a dummy (noise) feature with random generated values in training data. Our 1st hypothesis was that if there are features whose influence on the model is less than that of this random variable, they can be removed without loss, and perhaps even with an improvement in model performance.


```
X['random'] = np.random.normal(0, 1, size = X.shape[0])
```


<p align="center">
  <img src="https://github.com/veidlink/Hackathon-of-the-Graduate-School-of-Business-and-VK/assets/137414808/e0082d63-e0bd-49ff-b896-540c835801b7" alt="Description of Image">
</p>


From the provided visualization, it can be understood that all features are relatively important for classification as their shap values are greater than that of a random feature. We also tried to exclude some features by threshold shap value, but in each case this only brought a loss of quality. For this reason, we left all features except 5th. 

The next problem we tried to deal with was class imbalance. In the training model, the positive class is represented about half as much as the negative class. This is not a critical imbalance, but it hinders the potential of classification models.

<p align="center">
  <img src="https://github.com/veidlink/Hackathon-of-the-Graduate-School-of-Business-and-VK/assets/137414808/f0e53f2e-700f-4bd0-929c-53e2fe32f0d2" alt="Description of Image">
</p>

We tried using the SMOTE method (Synthetic Minority Over-sampling Technique) to tackle this issue. The main idea of SMOTE is to create synthetic (not real) samples from the class with fewer observations to balance the class distribution. For each selected sample from the minority class, the method finds its k nearest neighbors, takes the selected sample and one of its nearest neighbors, and then creates a synthetic sample lying on the line connecting these two samples. This is done by choosing a random point on this line. The process is repeated until the number of samples in the minority class becomes comparable to the number of samples in the majority class.

```
from imblearn.over_sampling import SMOTE

# Apply SMOTE oversampling to the training data
sm = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = sm.fit_resample(X, y)
```


<p align="center">
  <img src="https://github.com/veidlink/Hackathon-of-the-Graduate-School-of-Business-and-VK/assets/137414808/47ce0bde-1a4b-4aac-bcf7-a786db99987c" alt="Description of Image">
</p>


However, this idea, due to non-linear relationships in the data, did not improve the result. In fact, the metric dropped by about 0.01. Among the hypotheses we tested were also dimensionality reduction and data clustering (clusters participated in the prediction as an additional feature). All of that wasn't effective either. We also tried to write a custom fully-connected multi-layer neural network, but it turned out to be too complex. 


To sum up, CatBoost gave us a score of about 0.85 on the f1-score metric, but we didn't stop there.

```
CatBoost_best_params = {'depth': 8,
 'learning_rate': 0.14993163898007728,
 'iterations': 807,
 'l2_leaf_reg': 2.0883307004683904,
 'min_data_in_leaf': 3,
 'bagging_temperature': 0.5717641539824976}
```

### 📝 Our best solution | Task 1

Our most prodctive approach was a pipeline of a StandardScaler and a Multi-layer Perceptron neural network tuned through GridSearchCV. It yielded us the finest result of F1-score equal to 0.9213. 
We used the following parameters:

```
{'mlpclassifier__activation': 'relu',
 'mlpclassifier__alpha': 0.05,
 'mlpclassifier__hidden_layer_sizes': (120, 60),
 'mlpclassifier__learning_rate': 'constant',
 'mlpclassifier__solver': 'adam'}
```

### ⚙️  Guide for MLP Classifier Inference Script

Firstly, clone this repossitory, navigate to the folder and run this line of code to install all requirements:


```
pip install -r requirements.txt
```

To use out model, ensure you're in the folder containing *TASK1-tune-fit-predict.py*. Run the script by typing:

```
python TASK1-tune-fit-predict.py
``` 


To tune the model yourself and find the best parameters pass _--tune_ as an argument or stick with --no-tune to use our best settings. The script will save the predictions in a file named results.csv in the same folder.

---


## 🎯 Task 2 | Image recognition for sanctuary by computer vision models

A large nature reserve is conducting a tender to develop an AI service to assist in tracking wildlife populations, aiming to relieve staff from the current manual monthly counting process. VK company is interested in securing this major contract, promising a bonus equivalent to one month's salary for the team that presents a machine learning model accurately and quickly recognizing elements in images. The task involves counting the number of squares in a set of test images containing geometric shapes like squares, rectangles, parallelograms, and circles. The dataset includes images with only squares, images with squares and other shapes without overlaps, and images with squares and overlapping shapes. The training set (train.csv) provides the image path, the number of squares, and the image type, while the test set (test.csv) lacks square count and type. Model's performance was evaluated using the Root Mean Square Error (RMSE) metric.

---

## 🎯 Task 3 | Recommendation system for advertising banners

myTarget is a self-service advertising platform for social networks such as VKontakte and Odnoklassniki, as well as other VK projects, covering over 90% of Russian-speaking internet users. Owned by VK, the platform prioritizes user experience by continually refining algorithms to display only relevant advertisements.

With the upcoming holiday season, major marketing companies task the myTarget team with building an improved banner recommendation system based on view and like logs. The logs include user_id (user identifier), item_id (banner identifier), like (whether the user liked the banner), and timestamp (Unix time in seconds of the action). Additionally, users and banners have features with a dimensionality of 32.

Our goal was to predict 20 banners for users, and the solution's quality was evaluated based on the proportion of "liked" banners by users from your proposed list (top-20 accuracy).

### Tried approaches | Task 3

- **SVD of User-Item Matrix**: filling missing values with -2 and use low-rank factorizations ($r = 20$). 
- **User-Based Approach (Collaborative Filtering)**: Calculate similarities between users and recommend banners liked by similar users for each user..
- **Item-Based Approach**: recommend for each user banners similar to ones he liked. Our best solution (0.535 top-20 accuracy)
- **Clustering**: recommend banners from the most liked category (did not work because clusters are not large enough).

More details about EDA and our implementations of algorithms above in `.ipybn` file in `Task3` folder.

### ⚙️ Tech stack 
- **Scikit-learn** 
- **Numpy** 
- **Scipy**
- **Pandas**
- **Seaborn** for visualization
- **Yellowbrick** for clustering
- **Catboost** for classification approach.
  
  You can also run a `task3_script.py` *having all test and train data in the same directory* to get best predictions according to item-based approach. 

---

<p align="center">
  <img src="https://github.com/veidlink/Hackathon-of-the-Graduate-School-of-Business-and-VK/assets/137414808/077b38da-ed84-47b2-ba16-5fa8f2ad3211" alt="Description of Image">
</p>
