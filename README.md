# nfmodelapis
Central repo for all model training, and pipeline tools for usage with all data collection APIs from nfflow.


Example:

### Unsupervised Learning
```python
from nfmodelapis.text.SentenceEmbedder import ModelSelect
trainer = ModelSelect(model_name,
                      model_output_path,
                      model_architecture=model_architecture
                       ).return_trainer()
trainer.train(data_path=os.path.join(
                save_timestamp,json_filename))
```
### Question Answering

```python
from nfmodelapis.text.question_answering import QAPipeline
pipe = QAPipeline(final_df)
res = pipe.batch_qa(qa_query, column_name)  #column name in df for performing the question answering
print(res)
```
### Summarisation

```python
from nfmodelapis.text.summarization import SummarizationPipeline
pipe = SummarizationPipeline(final_df)
res = pipe.batch_summarize(column_name)  #column name in df for performing the summarisation
```

### Entity Extraction

```python
from nfmodelapis.text.ner import NERPipeline
ner = NERPipeline(df)
ents = ner.batch_ner(column_name)  #column name in df for performing the entity extraction
```


