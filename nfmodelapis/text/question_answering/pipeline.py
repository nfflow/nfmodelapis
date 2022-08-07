from transformers import pipeline
import pandas as pd


class QAPipeline:
    def __init__(self, model, tokenizer, data):
        """

        Parameters
        ----------
        model : str or Model Object
                String containing model path or
                Model object from huggingface
        tokenizer : str or Model Object
            String containing tokenizer path or
            Tokenizer object from huggingface.
        data : str or pandas DataFrame
            string containing path to dataframe or pandas DataFrame object.

        Returns
        -------
        None.

        """
        self.model = model
        self.tokenizer = tokenizer
        if type(data) == str:
            if data.endswith('.csv'):
                df = pd.read_csv(data)
            elif data.endswith('.json'):
                df = pd.read_json(data)
        else:
            df = data
        self.df = df

    def check_transformers_installation(self):
        try:
            import transformers
            transformers.__init__
        except ImportError:
            raise('''Transformers not installed.
                  Install Transformers via `pip install transformers''')

    def question_answering(self, question, context):
        self.check_transformers_installation()
        pipe = pipeline('question-answering',
                        model=self.model,
                        tokenizer=self.tokenizer)
        res = pipe(question, context)
        return res
