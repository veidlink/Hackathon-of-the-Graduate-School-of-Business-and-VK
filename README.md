# Hackathon-of-the-Graduate-School-of-Business-and-VK
Predicting patient recovery using ML algorithms, image recognition for sanctuary by computer vision models, recommendation system for advertising banners 

## 🦸‍♂️ Team
- [Solomon](https://github.com/veidlink)
- [Roman]()
- [Katerina]()
- [Nikita]()

## 🎯 Task 1


The task for VK Predict, a business unit of the company VK, involves developing a predictive model for a pharmaceutical company. This model should predict patient recovery based on test results, specifically for those treated with a new, expensive medication designed to replace a less effective, cheaper one. The data consists of anonymized patient features correlating to test results, with a training dataset (train.csv) including a target feature indicating whether the illness was cured by the medication (values 0 or 1), and a test dataset (test.csv) for predictions. Solution was evaluated using the F1-score metric.

## Feature selection


We dropped feature number 5 because it was highly linearly correlated with the target variable in the training dataset, while this was not observed in the test dataset. The model was overfitting on the 5th feature, which was causing a decrease in performance, and we corrected this

|   | target |
|---|--------|
| 5 | 1.00000|
| target | 0.85876|

After obtaining the initial result with the CatBoost model tuned using the Optuna library, we visualized the feature importances to test several hypotheses. 

Our 1st hypothesis was that if there are features whose influence on the model is less than that of this random variable, they can be removed without loss, and perhaps even with an improvement in model performance.


```
X['random'] = np.random.normal(0, 1, size = X.shape[0])
```


![image](https://github.com/veidlink/Hackathon-of-the-Graduate-School-of-Business-and-VK/assets/137414808/a9ef88f7-a40c-4eb8-96fe-f91d76c4efd3)


From the provided visualization, it can be understood that all features are relatively important for classification as their shap values are greater than that of a random feature. We also tried to exclude some features by threshold shap value, but in each case this only brought a loss of quality. For this reason, we left all features except 5th. 

The next problem we tried to deal with was class imbalance. In the training model, the positive class is represented about half as much as the negative class. This is not a critical imbalance, but it hinders the potential of classification models.

![image](https://github.com/veidlink/Hackathon-of-the-Graduate-School-of-Business-and-VK/assets/137414808/b87592e5-8255-4a4d-8812-de0e349e96cb)

We tried using the SMOTE method (Synthetic Minority Over-sampling Technique) to tackle this issue. The main idea of SMOTE is to create synthetic (not real) samples from the class with fewer observations to balance the class distribution. For each selected sample from the minority class, the method finds its k nearest neighbors, takes the selected sample and one of its nearest neighbors, and then creates a synthetic sample lying on the line connecting these two samples. This is done by choosing a random point on this line. The process is repeated until the number of samples in the minority class becomes comparable to the number of samples in the majority class

```
from imblearn.over_sampling import SMOTE

# Apply SMOTE oversampling to the training data
sm = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = sm.fit_resample(X, y)
```


![image](https://github.com/veidlink/Hackathon-of-the-Graduate-School-of-Business-and-VK/assets/137414808/4a2070c0-f9ac-4d55-8baa-a68666df3d40)


However, this idea, due to non-linear relationships in the data, did not improve the result. In fact, the metric dropped by about 0.01. Среди гипотез, которые мы проверяли, также были понижение размерности и кластеризация данных (кластеры участвовали в предсказании как дополнительный признак).


To sum up, CatBoost gave us a score of about 0.85 on the f1-score metric, but we didn't stop there.


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

![image](https://github.com/veidlink/Hackathon-of-the-Graduate-School-of-Business-and-VK/assets/137414808/a046a619-5f67-4fd2-96a2-4e3800ed89bb)

