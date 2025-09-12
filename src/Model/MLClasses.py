from DataClasses import GenericData

class MLModel:
    input_data = GenericData
    output_data

    def __init__(self):
        pass

    def preproc(self):
        pass

    def run_model(self):
        pass

    def set_input_data(self, input_data):
        self.input_data = input_data

    def generate_output(self):
        self.output_data = self.run_model()

    def get_output_data(self):
        return self.output_data

