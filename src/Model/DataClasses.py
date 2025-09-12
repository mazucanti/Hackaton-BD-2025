

class GenericData:
    dataframe
    filepath = str
    encoding = str
    
    def __init__(self, filepath, encoding):
        self.filepath = filepath
        self.encoding = encoding

    def read_file(self):
        pass
    
    def write_file(self, path):
        pass
    
    def data_cleanup(self):
        pass

    def data_summary(self):
        pass

class SKU(GenericData):
    pass

class PDV(GenericData):
    pass

class Transactions(GenericData):
    pass
