# Train a BERT Encoder on Structured Electronic Health Record Data

## Description
This package allows to train and fine-tune an EHR records-based BERT model that can be employed for disease prediction as a downstream task. The pre-training and fine-tuning procedure most closely follows the method used in [Med-BERT](https://doi.org/10.1038/s41746-021-00455-y). 

_While the utilities of this repo are well equipped for several downstream clinical tasks, you may notice type 1 diabetes (T1D) risk metrics that were used as an example to build up this work_

## Installation
### 1. Obtain the data in BERT Format

Two data files are needed for the training to start:  
1. Transformed EHR data saved as a `parquet` file. 
2. Model vocabulary in `txt` format

The input data is assumed to have the following schema: person_id (int), sorted_event_tokens (array), day_position_tokens (array). Refer to `src/datasets/dummy_data.parquet` to view the expected data format:

- person_id: A unique identifier for each individual.
- day_position_tokens: An array representing the relative time (in days) of events.
- sorted_event_tokens: A list of event codes associated with the individual. Each event corresponds to the relative date indicated by its index in the day_position_tokens array.  
**Important** The first tokens are always assumed to be demographic tokens, e.g. age, ethnicity, gender or race with day_position_tokens set at 0, the remaining tokens should follow from 1.
- label: Label for the patient for a specific prediction task.

### 2. Clone the repo 
```
git clone {url-to-sanofi-clinical-bert-training_repo.git}
```

### 3. Create an environment for the project
```
conda create -n clinicalberttraining python=3.10 -y
```

### 4. Activate the environment
```
conda activate clinicalberttraining
```

### 5. Install dependencies
``` 
# Navigate to the Clinical-BERT-Training directory
pip install -r requirements.txt
```

___________

##  BERT Usage

Navigate to `src` directory to run the pipelines.
```
cd src/
```

Args:
* `-c` or `--config-path`
  * Path to the BERT configuration file.

### Configuration
1. Place the data files in `datasets/bert/` folder
2. Explore the config files in the `/configs` folder inside `src`
3. Provide the paths to the data files in the appropriate sections of a config file - `pretrain.yaml` for a pretraining task, and `bert.json` or `bert_reg.json` for a fine-tuning task
4. For pretraining task consider changing `eval_steps` and `save_steps` to a larger number in case of a large dataset. Otherwise, it is recommended to **not** change any other parameters, unless you know what you are doing.
5. In case of exploding gradients during training, slightly reduce the starting `learning_rate`
6. **Important** Keep in mind that depending on the available compute resources and the size of the dataset the initial pre-training might take up to several weeks!


### Launch
Run using the below commands depending on your initial task:

##### Pretraining
```
python pretrain_bert.py -c configs/pretrain.yaml
```

##### Fine-tuning
```
python finetune_bert.py -c configs/bert.json
```

##### Inference

```
python run_bert_inference.py -c configs/bert.json
```

____________

### Contacts
For any inquiries please raise a git issue and we will try to follow-up in a timely manner.

### Acknowledgements
This project follows the logic and original configuration developed by the authors of [Sanofi Clinical BERT Explainability](https://github.com/Sanofi-Public/Clinical-BERT-Explainability).
