import polars as pl

class GenericData:
    dataframe
    filepath: str
    encoding: str
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.encoding = encoding

    def read_file(self, filepath):
        return pl.scan_parquet(filepath)
    
    def write_file(self, path):
        pass
    
    def data_cleanup(self):
        pass

    def data_summary(self):
        pass

class SKU(GenericData):
    pass

class PDV(GenericData):

    def __init__(self, filepath, encoding):
        super.__init__(filepath, encoding)
        dataframe = self.read_file(self.filepath)


    def _prepare_pdv(pdv: pl.LazyFrame, transactions: pl.LazyFrame) -> pl.DataFrame:
        return dataframe.with_columns(
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
            'transaction_date',
            'reference_date',
            'premise',
        ).group_by('pdv').agg(
            pl.col('quantity', 'gross_value', 'net_value', 'gross_profit', 'discount', 'taxes').mean(),
            pl.col('zipcode', 'categoria_pdv').implode().first(),
            pl.col('distributor_id').n_unique().alias('distributor_count'),
            pl.col('gross_profit').len().alias('transaction_count'),
            pl.col('internal_product_id').implode().alias('transacted_products'),
        ).collect()

    def data_cleanup(self):
        return _prepare_pdv()

class Transactions(GenericData):
    pass
