import polars as pl
import polars.selectors as cs
import numpy as np

from src.Model.DataClasses import GenericData
from sklearn.cluster import HDBSCAN
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer, OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline

class MLModel:
    input_data: GenericData

    def __init__(self, data):
        self.set_input_data(data)

    def preproc(self):
        pass
    
    def param_search(self):
        pass

    def fit_transform(self):
        pass

    def set_input_data(self, input_data: GenericData):
        self.input_data = input_data

class ThompsonSampling():
    transactions: pl.LazyFrame
    stores: pl.LazyFrame

    def __init__(self, trans_path, store_path):
        self.transactions = pl.scan_parquet(trans_path)
        self.stores = pl.scan_parquet(store_path)

    def _get_sample_list(self, ):
        self.samples = [np.random.beta(s+1, l+1) for s, l in self.succ_loss]

    def _update_best_sku(self):

        if np.random.uniform() < self.probabilities[self.best_sku]:
            self.succ_loss[best_sku][0] += 1
        else:
            self.succ_loss[best_sku][1] += 1
        
    def _get_best_sku(self):
        self.best_sku = np.argmax(samples)

    def preproc(self) -> pl.LazyFrame:
        return self.stores.join(
            self.transactions, how='inner', left_on='pdv', right_on='internal_store_id'
            ).select(
                'pdv',
                'categoria_pdv',
                'transaction_date',
                'quantity',
                'internal_product_id'
            ).filter(
                pl.col('quantity') > 0
            ).with_columns(
                pl.col('transaction_date').dt.week().alias('week'),
            ).group_by(
                ['pdv', 'week', 'internal_product_id']
            ).agg(
                pl.col('quantity').sum(),
                pl.col('categoria_pdv').first()
            ).group_by(
                ['categoria_pdv', 'internal_product_id', 'week']
            ).agg(
                pl.col('quantity').mean()
            ).sort('week')

    