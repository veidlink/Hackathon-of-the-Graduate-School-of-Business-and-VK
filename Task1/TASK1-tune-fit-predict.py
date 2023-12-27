import warnings
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
import argparse

def remove_outliers(df, col):
    q75, q25 = np.percentile(df[col], [75, 25])
    iqr = q75 - q25
    upper_bound = q75 + 1.5 * iqr
    lower_bound = q25 - 1.5 * iqr
    return df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

def main(tune):
    df = pd.read_csv('./train.csv')
    data_test = pd.read_csv('./test.csv')

    X, y = df.drop(['5', 'target'], axis=1), df['target']
    X_test = data_test.drop(['5'], axis=1)

    for column in df.columns[:-2]:
        df = remove_outliers(df, column)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mlp_clf = make_pipeline(StandardScaler(), MLPClassifier(random_state=42))

    if tune:
        mlp_param_grid = {
            'mlpclassifier__hidden_layer_sizes': [(16, 8), (60, 30), (120, 60)],
            'mlpclassifier__activation': ['relu', 'tanh', 'logistic'],
            'mlpclassifier__solver': ['sgd', 'adam'],
            'mlpclassifier__alpha': [0.0001, 0.05],
            'mlpclassifier__learning_rate': ['constant', 'adaptive']
        }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grid_mlp = GridSearchCV(estimator=mlp_clf, param_grid=mlp_param_grid, n_jobs=-1, cv=kf)
            grid_mlp.fit(X, y)
            best_params = grid_mlp.best_params_

        preds_test = pd.DataFrame(grid_mlp.best_estimator_.predict(X_test)).rename({0: 'target'}, axis=1)
        preds_test.to_csv('./results.csv', index=False)   

        print(f'Model tuned, pedictions made.\nYour parameters: {best_params}')

    else:
        MLP_best_params = {
            'activation': 'relu',
            'alpha': 0.05,
            'hidden_layer_sizes': (120, 60),
            'learning_rate': 'constant',
            'solver': 'adam'
        }

        pipeline = make_pipeline(
            StandardScaler(),
            MLPClassifier(**MLP_best_params, random_state=42)
        )

        pipeline.fit(X, y)

        preds_test = pd.DataFrame(pipeline.predict(X_test)).rename({0: 'target'}, axis=1)
        preds_test.to_csv('./results.csv', index=False)
    
        print(f'Pedictions made.\nDefault parameters used: {MLP_best_params}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MLP Classifier with or without tuning.")
    
    parser.add_argument("--tune", action='store_true', help="Set this flag to tune the model.")
    parser.add_argument("--no-tune", action='store_false', dest='tune', help="Set this flag to use default parameters. This is the default behavior.")

    args = parser.parse_args()
    main(tune=args.tune)
