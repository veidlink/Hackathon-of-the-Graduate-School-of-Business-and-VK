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
Мы дропнули 5 фичу, ибо она сильно линейно коррелировала с целевой переменной в обучающей выборке, в то же время этого не наблюдалось на тестовой выборке. Модель переобучалась на 5 признак, соответственно качество падало, что мы и исправили.

|   | target |
|---|--------|
| 5 | 1.00000|
| target | 0.85876|


### 📝 Solution | Task 1
Our final solution was tuned perceptron neural network with these parameters: 
```
{'mlpclassifier__activation': 'relu',
 'mlpclassifier__alpha': 0.05,
 'mlpclassifier__hidden_layer_sizes': (60, 30),
 'mlpclassifier__learning_rate': 'constant',
 'mlpclassifier__solver': 'adam'}
```
