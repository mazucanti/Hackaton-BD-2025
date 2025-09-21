import polars as pl
import polars.selectors as cs

from src.Model.xgb import MixedModel
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from xgboost.sklearn import XGBRegressor


def scan_parquets(sku_file, transactions_file, pdv_file):
    sku = pl.scan_parquet(sku_file)
    transacoes = pl.scan_parquet(transactions_file).filter(pl.col('quantity')>0)
    pdv = pl.scan_parquet(pdv_file).join(
        transacoes, left_on='pdv', right_on='internal_store_id', how='semi'
    ).collect().lazy()

    transacoes = transacoes.join(pdv, left_on='internal_store_id', right_on='pdv', how='semi').collect().lazy()

    return sku, transacoes, pdv

def group_transactions_by_week_and(grouping_column, transactions: pl.LazyFrame, stores: pl.LazyFrame, profile=False) -> pl.DataFrame:
    both = ['internal_store_id', 'categoria_pdv']
    both.remove(grouping_column)
    other_column = both[0]
    ## tabela de transações possui pdvs que não estão na tabela de pdv
    tmp = (
        transactions.join(
                stores, how='inner', left_on='internal_store_id', right_on='pdv'
            ).with_columns(
                week=pl.col('transaction_date').dt.week(),
                premise=pl.when(pl.col('premise')=='On Premise').then(1).otherwise(0),
            ).cast({
                'internal_store_id':pl.Categorical,
                'internal_product_id':pl.Categorical,
                'categoria_pdv':pl.Categorical,
            }).drop(
                'transaction_date', 'reference_date', 'distributor_id', 'zipcode', other_column
            )
    )
    if grouping_column == 'internal_store_id':
        tmp = tmp.group_by(['internal_store_id', 'week']).agg(
                    pl.col('internal_product_id').implode(),
                    pl.col('cluster', 'premise').first(),
                    pl.col("quantity", "gross_value", "net_value", "gross_profit", "discount", "taxes").sum(),
                )
    elif grouping_column == 'categoria_pdv':
        tmp = tmp.group_by(['categoria_pdv', 'week']).agg(
                    pl.col('internal_product_id').implode(),
                    pl.col('cluster', 'premise').first(),
                    pl.col("quantity", "gross_value", "net_value", "gross_profit", "discount", "taxes").mean(),
                )
    else: raise ValueError

    return tmp.with_columns(
                    pl.col('internal_product_id').list.len().alias('product_diversity'),
                    pl.col('internal_product_id').list.eval(pl.element().mode()).list.first().alias('most_bought_product')
                ).drop('internal_product_id').sort(
                    [grouping_column, 'week']
                ).profile(show_plot=profile)[0]

def prepare_df_for_xgb(transactions: pl.LazyFrame | pl.DataFrame, period_to_train=8, profile=False, grouping_column='internal_store_id', benchmark=False) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame | None]:
    transactions = transactions.lazy()
    ## dictionary containing pdv: most common product (after grouping by week)
    product_by_store = dict(
        transactions.group_by(grouping_column).agg(
            pl.col('most_bought_product').mode().first()
        ).collect().iter_rows()
    )
    columns_to_window = ['quantity', 'gross_value', 'net_value', 'gross_profit', 'discount', 'taxes', 'product_diversity']
    df_period = period_to_train + 4 ## 4 weeks_to_predict
    temp = (
        ## create dataframe with the cartesian product of pdv's and 1-52 weeks
        pl.LazyFrame({
            grouping_column: product_by_store.keys(),
            'week': [list(range(1,53))]*len(product_by_store)
        }).cast({
            grouping_column:pl.Categorical
        }).explode('week')
        ## now, retrieve the transaction information of those pdv's and weeks
        .join(
            transactions.drop('most_bought_product'), on=[grouping_column, 'week'], how='left', validate='1:1'
        )
        ## the weeks that have no transaction will have nulls, so
        .fill_null(0)
        ## let's make week-windows of size df_period (note que interval is closed on both ends) to train the NN
        .group_by_dynamic(
            index_column='week', every='1i', period=f'{df_period-1}i', group_by=grouping_column, closed='both'
        ).agg(pl.all())
        ## the last lines will be windows smaller than df_period, so we remove them
        .filter(
            pl.col('quantity').list.len()==df_period
        )
        ## the windows are in list form; we now explode them horizontally, taking care of the names
        .with_columns(
            (pl.col(list_column).list.to_struct(
                fields=lambda k : f'{list_column}{k+1:02d}', upper_bound=df_period
            ).struct.unnest() for list_column in columns_to_window)
        ).with_columns(
            pl.col(col).list.first().alias(col) for col in ['cluster', 'premise']
        )
        ## drop the list columns, the pdv column and, since we want to predict the quantity, drop the four last columns of the other variables that were unnested
        .drop(
            columns_to_window
        )
        ## correct the week column to be the week end of each window; this ensures the last window have 'week' as 52
        .with_columns(
            pl.col('week')+df_period-1
        ).profile(show_plot=profile)[0]
    )
    if not benchmark:
        X_pred = temp.clone()
    else:
        X_pred = None
    training_data = temp.drop(
        ((~cs.starts_with('quantity')) & cs.ends_with(*list(map(lambda x: f'{x:02d}', range(period_to_train+1, period_to_train+5))))),
        grouping_column
        )
    y_columns = list(map(lambda x: f'quantity{x:02d}', range(period_to_train+1, period_to_train+5)))
    return training_data.drop(y_columns), training_data.select(['week']+y_columns), X_pred

def functionalize_dataset_ranges(X, y, columns_to_train):
    assert 'quantity' in columns_to_train, 'It would be wise to use previous quantities to predict future quantities.'
    X_eq = lambda k : X.filter(pl.col('week')==k).drop('week').select(cs.starts_with(*columns_to_train))
    y_eq = lambda k : y.filter(pl.col('week')==k).drop('week')
    return X_eq, y_eq

def train_by_week(X, y, X_pred, test_last_weeks: list[int], training_period, columns_to_train, regressor_info: dict, classifier_info=None, bench=False):
    X_eq, y_eq = functionalize_dataset_ranges(X, y, columns_to_train)
    n_variables = X_eq(test_last_weeks[0]).shape[1]
    estimator = MixedModel(n_variables, regressor_info, classifier_info, mix_model=True)

    for week in test_last_weeks:
        if bench:
            print('# week:', week)
            fit_and_score(
                estimator, week, X_train=X_eq(week-4), y_train=y_eq(week-4), X_test=X_eq(week), y_test=y_eq(week), period=training_period
            )
        else:
            assert len(test_last_weeks)==1, 'You sould pass only the final week of the training period.'

            estimator.fit(
                X_eq(week), y_eq(week)
            )
            columns = X_eq(week).columns
            pdvs = X_pred.filter(pl.col('week')==week).drop('week').select('internal_store_id')
            X_pred = X_pred.drop(cs.ends_with('1', '2', '3', '4'))
            for k in range(5, 5+training_period):
                X_pred = X_pred.with_columns(
                    cs.ends_with(f'{k:02d}').name.map(lambda text:text.removesuffix(f'{k:02d}')+f'{k-4:02d}')
                )
            return pl.concat([
                pdvs,
                pl.DataFrame(estimator.predict(X_pred.filter(pl.col('week')==week).select(columns)))
            ], how='horizontal')

def fit_and_score(estimator: MixedModel, week, X_train: pl.DataFrame, y_train, X_test, y_test, period):
    (n_samples, n_variables), variable_names, output_names = X_train.shape, X_train.columns, y_train.columns
    print((n_samples, n_variables), variable_names, output_names)

    estimator.fit(X_train, y_train)
    print(
        f'MAE no treino: {estimator.score(X_train, y_train):.2f} treinando com semana {week-8-period+1} a semana {week-8} prevendo {week-7} a {week-4}')
    print(
        f'MAPE do anterior: {estimator.MAPrediction(X_train, y_train, multioutput='uniform_average', debug=False)}')
    print('--------------')
    print(
        f'MAE no teste {estimator.score(X_test, y_test):.2f} modelo acima prevendo {week-3} a {week}')
    print(
        f'MAPE do anterior: {estimator.MAPrediction(X_test, y_test, multioutput='uniform_average', debug=False)}')
    print('-----------------------')

def make_multi_classifier():
    return MultiOutputClassifier(HistGradientBoostingClassifier())
def make_multi_regressor():
    return MultiOutputRegressor(XGBRegressor())

def train_model(period, transactions, clusterized_pdv, final_weeks, benchmark=False):
    print('------------------------------------------------------------------------------------------------------------------------------\nperiod size:', period)
    grouping_column = 'internal_store_id'
    preprocessed = group_transactions_by_week_and(grouping_column, transactions, clusterized_pdv)
    X, y, X_pred = prepare_df_for_xgb(preprocessed, period_to_train=period, grouping_column=grouping_column, benchmark=benchmark)

    regressor_info = {
        'Model': make_multi_regressor,
        'scaler': StandardScaler,
        'params': {
            'estimator__booster': 'dart',
            'estimator__colsample_bylevel': 0.6793,
            'estimator__colsample_bynode': 0.918,
            'estimator__colsample_bytree': 0.492,
            'estimator__eval_metric': 'rmse',
            'estimator__gamma': 2.988191101963424,
            'estimator__importance_type': 'gain',
            'estimator__learning_rate': 0.305,
            'estimator__max_delta_step': 4.12,
            'estimator__max_depth': 4,
            'estimator__max_leaves': 296,
            'estimator__n_estimators': 300,
            'estimator__reg_alpha': 1.92,
            'estimator__reg_lambda': 1.86,
        }
    }
    classifier_info = {
        'Model': make_multi_classifier,
        'scaler': StandardScaler,
        'threshold': 0.9,
        'params': {
            'estimator__l2_regularization': 0.2,
            'estimator__learning_rate': 0.219,
            'estimator__max_bins': 16,
            'estimator__max_depth': 10,
            'estimator__max_features': 0.1,
            'estimator__max_iter': 50,
            'estimator__max_leaf_nodes': 81,
            'estimator__min_samples_leaf': 100,
            'estimator__validation_fraction': 0.1,
            'estimator__categorical_features': [0, 1],
        }
    }
    columns_to_train = [
            'quantity',
            'cluster',
            'gross_value',
            'net_value',
            'gross_profit',
            'discount',
            'taxes',
            'different_products',
            'premise',
        ]

    predictions = train_by_week(X, y, X_pred, final_weeks, training_period=period, columns_to_train=columns_to_train,
                regressor_info=regressor_info, classifier_info=classifier_info, bench=benchmark)
    if not benchmark:
        return predictions