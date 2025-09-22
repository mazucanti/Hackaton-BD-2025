import polars as pl
import polars.selectors as cs
import numpy as np


class ThompsonSampling:

    transactions: pl.LazyFrame
    stores: pl.LazyFrame
    prob: pl.LazyFrame

    def __init__(self, trans_path, store_path):
        self.transactions = pl.scan_parquet(trans_path)
        self.stores = pl.scan_parquet(store_path)

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

    def get_probabilities(self):
        raw_data = self.preproc()
        probability_table = raw_data.with_columns(
            alpha=1,
            beta=1,
            probability=0.0
        ).clone()

        probability_table = probability_table.group_by(
            ['categoria_pdv', 'internal_product_id']
        ).agg(
            pl.col('alpha', 'beta', 'probability').max()
        ).collect()

        # for test in
        for week in range(1,53):
            print(f'{week}/52')
            random_val = np.random.uniform()
                    
            probability_table = probability_table.lazy().join(
                raw_data.filter(pl.col('week') == week), on=['internal_product_id', 'categoria_pdv'], coalesce=True, how='left'
            ).with_columns(
                alpha=pl.when(pl.col('week').is_null()).then(
                    'alpha'
                ).otherwise(
                    pl.col('alpha') + 1
                ),
                beta=pl.when(pl.col('week').is_null()).then(
                    pl.col('beta') + 1
                ).otherwise(
                    'beta'
                )
            ).drop('week').with_columns(
                probability=pl.when(
                    pl.col('quantity').is_null()
                ).then(
                    'probability'
                ).otherwise(
                    pl.struct(
                        'alpha', 'beta'
                    ).map_elements(
                        lambda x: np.random.beta(x['alpha'], x['beta']), return_dtype=pl.Float64
                    )
                )
            ).drop('quantity').collect()
        self.prob = probability_table
    
    def scale_probs(self):
        return self.prob.lazy().group_by('categoria_pdv').agg(
            pl.col('internal_product_id', 'probability').implode()
        ).with_columns(
            scaled_prob=pl.col('probability').map_elements(lambda x: list(np.array(x)/sum(np.array(x))), return_dtype=list[float])
        ).join(
            self.stores, on='categoria_pdv', how='inner', validate='1:m'
            ).drop('premise', 'zipcode')