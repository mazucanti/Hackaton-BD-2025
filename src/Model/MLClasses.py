from DataClasses import GenericData
from sklearn.cluster import DBSCAN
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer, OrdinalEncoder, StandardScaler

class MLModel:
    input_data: GenericData
    output_data

    def __init__(self):
        pass

    def preproc(self):
        pass
    
    def param_search(self):
        pass
    
    def run_model(self):
        pass

    def set_input_data(self, input_data: GenericData):
        self.input_data = input_data

    def generate_output(self):
        self.output_data = self.run_model()

    def get_output_data(self):
        return self.output_data

class DBSCANPDV(MLModel):

    def __init__(self, pdvs: GenericData):
        super.__init__()
        self.set_input_data(pdvs)

    def preproc(self):
        non_scaled_pdv = self.input_data.data_cleanup()
        transformed_pdv = _apply_transformers_for_dbscan(non_scaled_pdv)

    def eps_search(self):
        
        with pl.Config(set_float_precision=1):
            for eps in np.arange(0.1, 2, 0.2):
                cluster_metrics_pdv(
                    pdv_with_cluster=train_dbscan_pdv(
                        transformed_pdv, eps=eps, min_samples=100
                    ),
                    non_scaled_df=non_scaled_pdv,
                    eps=eps
                )

    def _apply_transformers_for_dbscan(self) -> pl.DataFrame:

        transformer1 = ColumnTransformer(
            transformers=[('ord', OrdinalEncoder(), ['zipcode'])],
            remainder='passthrough',
            verbose_feature_names_out=False
        )
        transformer1.set_output(transform='polars')
        columns_scale = [coluna for coluna in self.input_data.columns if coluna not in ['pdv', 'zipcode']]
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
            transformer1.fit_transform(self.input_data)
        )

    def run_model(self):
        return train_dbscan_pdv(transformed_pdv, eps=0.9, min_samples=100)
    
    def train_dbscan_pdv(self, X_transformed: pl.DataFrame, eps: np.floating, min_samples: int) -> pl.DataFrame:
        numpy_X = X_transformed.drop('pdv').to_numpy()
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        return pl.concat([
            X_transformed,
            pl.DataFrame({'cluster':dbscan.fit_predict(numpy_X, sample_weight=None)})
            ], how='horizontal'
        )

    def cluster_metrics_pdv(pdv_with_cluster: pl.DataFrame, non_scaled_df: pl.DataFrame, eps: np.floating):
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

    
