from transformers import pipeline
import pandas as pd


class QAPipeline:
    def __init__(self, data, model=None, tokenizer=None):
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

    def qa(self, question, context):
        self.check_transformers_installation()
        model = self.model
        tokenizer = self.tokenizer
        if model and tokenizer:
            pipe = pipeline('question-answering',
                            model=self.model,
                            tokenizer=self.tokenizer)
        elif model:
            pipe = pipeline('question-answering',
                            model=self.model,
                            )
        elif tokenizer:
            pipe = pipeline('question-answering',
                            tokenizer=self.tokenizer)
        else:
            pipe = pipeline('question-answering',
                            )
        res = pipe(question, context)
        return res

    def batch_qa(self, question, context_col):
        data = self.df
        res_list = []
        for c in data[context_col]:
            if type(c) == str:
                res_list.append(self.qa(question, c))
        return res_list
