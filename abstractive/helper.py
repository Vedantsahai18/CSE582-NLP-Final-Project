import json
from datasets import Dataset, DatasetDict
import pandas as pd


def generate_train_dataset(dataset,tokenizer,train_size=1950,valid_size=400):
        max_input_length = 200
        max_target_length = 50

        def preprocess_function(examples):
            model_inputs = tokenizer(
                examples["document"],
                max_length=max_input_length,
                truncation=True
            )
            labels = tokenizer(
                examples["summary"], max_length=max_target_length, truncation=True
            )
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        train_data = dataset['test'].shuffle(seed=42).select(range(train_size)).map(preprocess_function, batched=True)
        validation_data = dataset['test'].shuffle(seed=42).select(range(valid_size)).map(preprocess_function, batched=True)
        return train_data,validation_data

def generate_test_dataset(dataset,tokenizer,test_size=650):
        max_input_length = 200
        max_target_length = 50

        def preprocess_function(examples):
            model_inputs = tokenizer(
                examples["document"],
                max_length=max_input_length,
                truncation=True
            )
            labels = tokenizer(
                examples["summary"], max_length=max_target_length, truncation=True
            )
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        test_data = dataset['test'].shuffle(seed=36).select(range(test_size)).map(preprocess_function, batched=True)
        return test_data

def write2file(location,score,dataset = "gigaword",dataset_type = 'validation'):
        with open(location+"/"+dataset+dataset_type+"_scores.json", 'w', encoding='utf-8') as f:
            json.dump(score, f, ensure_ascii=False, indent=4)
     

def convert_pd_to_gig(dataset2):
    """
    converts argilla dataset into gigaword format for testing
    """
    custom_data_dic = []

    #  loop over the dataframe
    for index, row in dataset2.iterrows():
        x = {'document': str(row['document']), 'summary': str(row['abs_summary'])}
        custom_data_dic.append(x)
    test_dataset = Dataset.from_list(custom_data_dic)
    # Create the DatasetDict
    dataset_dict = DatasetDict({
        "test": test_dataset,
    })

    return dataset_dict