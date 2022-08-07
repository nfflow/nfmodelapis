from transformers import pipeline


class QAPipeline:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

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
