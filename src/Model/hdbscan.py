import polars as pl
import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer, OrdinalEncoder, StandardScaler, QuantileTransformer


class HDBS:
    model: HDBSCAN
    prepared_data: pl.DataFrame
    transformed_X: pl.DataFrame

    def __init__(self, pdv, transactions, params=None) -> None:
        self.model = HDBSCAN()
        if params:
            self.model.set_params(**params)
        self.prepared_data = self._preparation(pdv, transactions)
        self.transformed_X = self._transform(self.prepared_data)

    def fit_predict(self) -> pl.DataFrame:
        numpy_X = self.transformed_X.drop('pdv').to_numpy()
        return pl.concat([
            self.prepared_data,
            pl.DataFrame({'cluster': self.model.fit_predict(numpy_X)})
            ], how='horizontal'
        )

    def _transform(self, prepared_pdv: pl.DataFrame) -> pl.DataFrame:
        columns_scale = [coluna for coluna in prepared_pdv.columns if coluna not in ['pdv', 'zipcode']]

        transformer1 = ColumnTransformer(
            transformers=[('ord', OrdinalEncoder(), ['zipcode'])],
            remainder='passthrough',
            verbose_feature_names_out=False
        ).set_output(transform='polars')

        transformer2 = ColumnTransformer(
            transformers=[
                ('scaled', StandardScaler(), ['zipcode']),
                ('power_transformed', StandardScaler(), columns_scale)
            ],
            remainder='passthrough',
            verbose_feature_names_out=False
        ).set_output(transform='polars')

        return transformer2.fit_transform(
            transformer1.fit_transform(prepared_pdv)
        )

    def _preparation(self, pdv: pl.LazyFrame, transactions: pl.LazyFrame) -> pl.DataFrame:
        return pdv.with_columns(
            pl.col('zipcode').cast(pl.String).name.keep(),
            premise_dummy=pl.when(pl.col('premise').str.contains('On')).then(1).otherwise(0),
        ).with_columns(
            pl.when(pl.col('zipcode') == '8107').then(pl.lit('08107')).otherwise('zipcode').name.keep(),
        ).join(
            transactions,
            left_on='pdv',
            right_on='internal_store_id',
            how='inner',
            validate="1:m"
        ).drop(
            'reference_date',
            'premise',
            'categoria_pdv', 'gross_value', 'gross_profit', 'discount', 'taxes'
        ).group_by('pdv').agg(
            pl.col('transaction_date').len(),
            pl.col('quantity', 'net_value').mean(),
            pl.col('zipcode').implode().list.first(),
            pl.col('distributor_id').n_unique().alias('distributor_count'),
            pl.col('quantity').len().alias('transaction_count'),
            # pl.col('internal_product_id').implode().alias('transacted_products'),
        ).collect()
    