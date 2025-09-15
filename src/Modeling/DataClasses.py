import polars as pl

class GenericData:
    lazyframe : pl.LazyFrame
    path_parquet : str
    encoding : str = 'UTF-8'
    
    def __init__(self, parquet_path: str, encoding: str) -> None:
        self.path_parquet = parquet_path
        self.encoding = encoding
        self.lazyframe = self.read_file()

    def read_file(self) -> pl.LazyFrame:
        return pl.scan_parquet(self.path_parquet)
    
    def write_file(self, path) -> None:
        self.lazyframe.sink_parquet(
            path, compression='lz4'
        )

    def data_summary(self) -> None:
        print(self.lazyframe.describe())

class SKU(GenericData):
    columns : list[str]


    def data_cleanup(self):
        self.lazyframe.drop('subcategoria')
        pass

class PDV(GenericData):
    columns : list[str]

    pass

class Transactions(GenericData):
    columns : list[str]

    pass
