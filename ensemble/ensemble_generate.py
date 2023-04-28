from transformers import AutoTokenizer
import os
import pandas as pd
import numpy as np
from ensemble_test import evaluate
from transformers import AutoModelForSeq2SeqLM
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def abstractive(path,text):
    # generate a summary for the test dataset
    tokenizer = AutoTokenizer.from_pretrained(path,local_files_only=True, cache_dir="model/")
    inputs = tokenizer(text, return_tensors="pt").input_ids
    model = AutoModelForSeq2SeqLM.from_pretrained(path)
    outputs = model.generate(inputs, max_new_tokens=15, do_sample=False)
    final_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return final_text

def extractive(path,text):
    
    ext_text = None
    return ext_text


def main():
  #loading the dataset
  df = pd.read_csv('corpusfinal.csv')

  #loading the model
  model_checkpoint = "t5-small"

  # generate a summary for the test dataset
  df_test = pd.DataFrame(columns=['gold_summary','modelsummary'])

  for index,row in df.iterrows():
    text = row["document"]
    summary = row["abs_summary"]
    print("Original Text : ")
    print(text)
    print("Extractive Summary : ")
    ext_text = extractive("model/",text)
    print(ext_text)
    print("Abstractive Summary : ")
    final_text = abstractive("model/",ext_text)
    print(final_text)
    print(25*"=")
    df_test = df_test.append({'gold_summary':summary,'modelsummary':final_text},ignore_index=True)

  # scores = evaluate( df_test, 0.6, 7)
  # print("ROGUE SCORES")
  # print(scores)

if __name__ == "__main__":
    main()
    