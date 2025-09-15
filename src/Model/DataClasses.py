import polars as pl

class GenericData:
    lazyframe: pl.LazyFrame
    filepath: str
    encoding: str
    
    def __init__(self, parquet_path: str) -> None:
        self.path_parquet = parquet_path
        self.lazyframe = self.read_file()

    def read_file(self) -> pl.LazyFrame:
        return pl.scan_parquet(self.path_parquet)
        
    def write_file(self, path) -> None:
        self.lazyframe.sink_parquet(
            path, compression='lz4'
        )

    
    def data_cleanup(self):
        pass

    def data_summary(self) -> None:
        print(self.lazyframe.describe())

class SKU(GenericData):
    pass

class PDV(GenericData):

    transactions: pl.LazyFrame

    def set_transactions(self, transactions):
        self.transactions = transactions

    def _prepare_pdv(self) -> pl.LazyFrame:
        return self.lazyframe.with_columns(
            pl.col('zipcode').cast(pl.String).name.keep(),
            premise_dummy=pl.when(pl.col('premise').str.contains('On')).then(1).otherwise(0),
        ).with_columns(
            pl.when(pl.col('zipcode') == '8107').then(pl.lit('08107')).otherwise('zipcode').name.keep(),
        ).join(
            self.transactions,
            left_on='pdv',
            right_on='internal_store_id',
            how='inner',
            validate="1:m"
        ).drop(
            'transaction_date',
            'reference_date',
            'premise',
        ).group_by('pdv').agg(
            pl.col('quantity', 'gross_value', 'net_value', 'gross_profit', 'discount', 'taxes').mean(),
            pl.col('zipcode', 'categoria_pdv').implode().list.first(),
            pl.col('distributor_id').n_unique().alias('distributor_count'),
            pl.col('gross_profit').len().alias('transaction_count'),
            pl.col('internal_product_id').implode().alias('transacted_products'),
        )

    def data_cleanup(self):
        self.lazyframe = self._prepare_pdv()

class Transactions(GenericData):
    pass
