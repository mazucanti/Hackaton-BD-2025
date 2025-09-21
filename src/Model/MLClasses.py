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

class DBSCANPDV(MLModel):

    def preproc(self):
        self.input_data.data_cleanup()
        self.input_data.lazyframe = self.input_data.lazyframe.drop(
            'transacted_products', 'categoria_pdv'
            )
        
        return self._apply_transformers_for_dbscan()

    def param_search(self):
        self.eps_search()
    
    def eps_search(self):
        with pl.Config(set_float_precision=1):
            for eps in np.arange(0.1, 2, 0.2):
                self.cluster_metrics_pdv(
                    pdv_with_cluster=train_dbscan_pdv(
                        self.preproc(), eps=eps, min_samples=100
                    ),
                    non_scaled_df=self.input_data.data_cleanup(),
                    eps=eps
                )

    def _apply_transformers_for_dbscan(self) -> pl.DataFrame:

        transformer1 = ColumnTransformer(
            transformers=[('ord', OrdinalEncoder(), ['zipcode'])],
            remainder='passthrough',
            verbose_feature_names_out=False
        )
        transformer1.set_output(transform='polars')
        columns_scale = [coluna for coluna in self.input_data.lazyframe.collect_schema() if coluna not in ['pdv', 'zipcode']]
        transformer2 = ColumnTransformer(
            transformers=[
                ('scaled', StandardScaler(), ['zipcode']),
                ('power_transformed', PowerTransformer(), columns_scale)
            ],
            remainder='passthrough',
            verbose_feature_names_out=False
        )
        transformer2.set_output(transform='polars')
        return transformer2.fit_transform(
            transformer1.fit_transform(self.input_data.lazyframe.collect())
        )

    def train_hdbscan_pdv(self, X_transformed: pl.DataFrame, cluster_size: tuple[int, int], min_samples: int) -> pl.DataFrame:
        numpy_X = X_transformed.drop('pdv').to_numpy()
        hdbscan = HDBSCAN(min_cluster_size=cluster_size[0], max_cluster_size=cluster_size[1], min_samples=min_samples, n_jobs=-1)
        return pl.concat([
            X_transformed,
            pl.DataFrame({'cluster':hdbscan.fit_predict(numpy_X)})
            ], how='horizontal'
        )
    
    def run_model(self, cluster_size):
        return self.train_hdbscan_pdv(
            self.preproc(),
            cluster_size,
            min_samples=100
            )

    def cluster_metrics_pdv(self, pdv_with_cluster: pl.DataFrame, non_scaled_df: pl.DataFrame, eps: np.floating):
        print(f''' Com eps {eps}, temos:
                {
                    pdv_with_cluster
                    .select('pdv', 'cluster')
                    .join(non_scaled_df, on='pdv', validate='1:1', maintain_order='right', how='full', suffix='_original')
                    .drop('pdv')
                    .group_by('cluster').agg(
                        (~cs.by_name('pdv_original', 'zipcode')).mean(),
                        pl.col('zipcode').mode().last(),
                        pl.col('pdv_original').len().alias('cluster_count')
                    )
                    .sort('cluster', descending=False)
                    .with_columns(
                        (pl.col('quantity')*pl.col('transaction_count')).alias('total_quantity')
                    )
                    .with_columns(
                        (pl.col('gross_value')*pl.col('total_quantity')).alias('total_gross_value')
                    )
                }\n\n
                '''
        )   
