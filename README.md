# nfmodelapis
Central repo for all model training, and pipeline tools for usage with all data collection APIs from nfflow.


Example:

```
from nfmodelapis.text.SentenceEmbedder import ModelSelect
trainer = ModelSelect(model_name,
                      model_output_path,
                      model_architecture=model_architecture
                       ).return_trainer()
trainer.train()
```
