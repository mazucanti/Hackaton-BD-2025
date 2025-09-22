import numpy as np
import polars as pl

from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, make_scorer, median_absolute_error
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from xgboost.sklearn import XGBRegressor

def MAPE_SCORE(y, pred):
    return np.mean(np.abs(y - pred)/np.maximum(y, np.ones_like(y)))

class MixedModel:
    classifier: Pipeline
    regressor: Pipeline
    regressor_cv: bool
    classifier_cv: bool
    threshold: float

    def __init__(self, n_variables, regressor_dict: dict, classifier_dict: dict, mix_model=False):
        if 'model_name' in regressor_dict:
            self.regressor_cv = True
            self.regressor = Pipeline([
                ('scaler', regressor_dict['scaler']()),
                ('model_instance', self.bayes_search(
                    regressor_dict['model_name'], n_variables
                ))])
        else:
            self.regressor_cv = False
            self.regressor = Pipeline([
                ('scaler', regressor_dict['scaler']()),
                ('model_instance', regressor_dict['Model']().set_params(
                    **(regressor_dict['params'])
                ))])
            
        if not mix_model: 
            self.classifier = None
            self.classifier_cv = False
            return

        self.threshold=classifier_dict['threshold']
        assert self.threshold <= 1
        assert self.threshold > 0

        if 'model_name' in classifier_dict:
            self.classifier_cv = True
            self.classifier = Pipeline([
                ('scaler', classifier_dict['scaler']()),
                ('model_instance', self.bayes_search(
                    classifier_dict['model_name'], n_variables
                ))])
        else:
            self.classifier_cv = False
            self.classifier = Pipeline([
                ('scaler', classifier_dict['scaler']()),
                ('model_instance', classifier_dict['Model']().set_params(
                    **(classifier_dict['params'])
                ))])
    
    def fit(self, X, y: pl.DataFrame):
        if self.classifier:
            self.classifier.fit(X, (y>0).cast(pl.Int8).to_numpy())
        self.regressor.fit(X, y.to_numpy())


    def predict(self, X):
        if self.classifier:
            tmp = np.array(self.classifier.predict_proba(X))[:,:,1].transpose()
            if self.threshold == 1:
                return np.multiply(
                    tmp == np.max(tmp),
                    # self.classifier.predict(X),
                    self.regressor.predict(X)
                )
            else:
                return np.multiply(
                    tmp > self.threshold,
                    self.regressor.predict(X)
                )
        else:
            return self.regressor.predict(X)
    
    def best_params_(self):
        params = {}
        if self.regressor_cv:
            params['regressor'] = self.regressor['model_instance'].best_params_
        if self.classifier_cv:
            params['classifier'] = self.classifier['model_instance'].best_params_

        return params
    
    def score(self, X, y, multioutput='uniform_average') -> float:
        print(f'MAE do regressor no abaixo: {mean_absolute_error(y.to_numpy(), self.regressor.predict(X))}')
        return mean_absolute_error(
            y, self.predict(X), multioutput=multioutput
        )
    
    def MAPrediction(self, X, y, multioutput='uniform_average', debug=False) -> float:
        prediction = self.predict(X)
        if prediction.sum() == 0.0:
            print('Modelo previu apenas nulos!')
        if debug:
            print(prediction)
            print(y.to_numpy())
        if multioutput=='uniform_average':
            return np.mean(self.MAPE(y.to_numpy(), prediction))
        else:
            return self.MAPE(y.to_numpy(), prediction)
        # return mean_absolute_percentage_error(
        #     y.to_numpy(), prediction, multioutput=multioutput
        # )
    
    def MAPE(self, y, pred):
        return np.mean(np.abs(y - pred)/np.maximum(y, np.ones_like(y)), axis=0)

    def bayes_search(self, model, n_variables):
        match model:
            case 'multi_xgb':
                return BayesSearchCV(
                    estimator=MultiOutputRegressor(XGBRegressor(
                        # missing=np.nan,
                        subsample=0.9,
                        random_state=411,
                        n_jobs=8,
                        objective='reg:squarederror',
                        booster='gbtree',
                    )),
                    search_spaces= {
                        # 'estimator__base_score': None,
                        # 'estimator__booster': Categorical(['gbtree', 'dart']),
                        # 'estimator__callbacks': None,
                        'estimator__colsample_bylevel': Real(0.4, 1),
                        'estimator__colsample_bynode': Real(0.4, 1),
                        'estimator__colsample_bytree': Real(0.4, 1),
                        # 'estimator__device': None,
                        # 'estimator__early_stopping_rounds': None,
                        # 'estimator__enable_categorical': False,
                        'estimator__eval_metric': Categorical(['mape', 'mphe', 'rmse']),
                        # 'estimator__feature_types': None,
                        # 'estimator__feature_weights': None,
                        'estimator__gamma': Real(0.01, 10, prior='log-uniform'),
                        # 'estimator__grow_policy': None,
                        'estimator__importance_type': Categorical(['gain', 'weight', 'cover', 'total_gain', 'total_cover']),
                        # 'estimator__interaction_constraints': None,
                        'estimator__learning_rate': Real(1e-4, 1, prior='log-uniform'),
                        # 'estimator__max_bin': None,
                        # 'estimator__max_cat_threshold': None,
                        # 'estimator__max_cat_to_onehot': None,
                        'estimator__max_delta_step': Real(0.01, 10, prior='log-uniform'),
                        'estimator__max_depth': Integer(2, 6),
                        'estimator__max_leaves': Integer(20, 300),
                        # 'estimator__min_child_weight': None,
                        # 'estimator__monotone_constraints': None,
                        # 'estimator__multi_strategy': None,
                        'estimator__n_estimators': Integer(20, 400),
                        # 'estimator__num_parallel_tree': None,
                        'estimator__reg_alpha': Real(0.1, 20),
                        'estimator__reg_lambda': Real(0.1, 2),
                        # 'estimator__sampling_method': None,
                        # 'estimator__scale_pos_weight': None,
                        # 'estimator__subsample': None,
                        # 'estimator__tree_method': None,
                        # 'estimator__validate_parameters': None,
                        # 'estimator__verbosity': 1,
                },
                n_iter=20,
                n_jobs=8,
                n_points=4,
                refit=True,
                verbose=0,
                random_state=411,
                scoring=make_scorer(median_absolute_error, greater_is_better=False))
            
            case 'multi_class': 
                return BayesSearchCV(
                estimator=MultiOutputClassifier(HistGradientBoostingClassifier(
                    loss='log_loss',
                    early_stopping=True,
                    verbose=0,
                    random_state=411,
                    scoring='accuracy'
                )),
                search_spaces={
                        'estimator__learning_rate': Real(0.001, 1, prior='log-uniform'),
                        'estimator__max_iter': Integer(50, 400),
                        'estimator__max_leaf_nodes': Integer(10, 100),
                        'estimator__max_depth': Integer(2, 10),
                        'estimator__min_samples_leaf': Integer(10, 100),
                        'estimator__l2_regularization': Real(0.1, 2),
                        'estimator__max_features': Real(0.1, 1.0),
                        'estimator__max_bins': Integer(16, 48),
                        # 'estimator__categorical_features': None,
                        'estimator__validation_fraction': Real(0.1, 0.9),
                },
                n_iter=20,
                n_jobs=4,
                n_points=4,
                refit=True,
                verbose=0,
                random_state=411,
            )