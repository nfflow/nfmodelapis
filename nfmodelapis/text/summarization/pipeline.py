from transformers import pipeline
import pandas as pd


class SummarizationPipeline:
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

    def summary(self, text):
        self.check_transformers_installation()
        model = self.model
        tokenizer = self.tokenizer
        if model and tokenizer:
            pipe = pipeline('summarization',
                            model=self.model,
                            tokenizer=self.tokenizer)
        elif model:
            pipe = pipeline('summarization',
                            model=self.model,
                            )
        elif tokenizer:
            pipe = pipeline('summarization',
                            tokenizer=self.tokenizer)
        else:
            pipe = pipeline('summarization',
                            )
        res = pipe(text)
        return res

    def batch_summarize(self, text_col):
        data = self.df
        res_list = []
        for text in data[text_col]:
            if type(text) == str:
                res_list.append(self.summary(text))
        return res_list
