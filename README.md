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

**1. Adding a random generated feature to see how it would affect the predictions of the model**


```
X['random'] = np.random.normal(0, 1, size = X.shape[0])
```


Our hypothesis was that if there are features whose influence on the model is less than that of this random variable, they can be removed without loss, and perhaps even with an improvement in model performance.
![image](https://github.com/veidlink/Hackathon-of-the-Graduate-School-of-Business-and-VK/assets/137414808/a9ef88f7-a40c-4eb8-96fe-f91d76c4efd3)


From the provided visualization, it can be understood that all features are relatively important for classification as their shap values are greater than that of a random feature.

### 📝 Solution | Task 1


Our best solution was a Multi-layer Perceptron neural network tuned through GridSearchCV with F1-score equal to 0.9213. 
We used the following parameters:

```
{'mlpclassifier__activation': 'relu',
 'mlpclassifier__alpha': 0.05,
 'mlpclassifier__hidden_layer_sizes': (120, 60),
 'mlpclassifier__learning_rate': 'constant',
 'mlpclassifier__solver': 'adam'}
```


