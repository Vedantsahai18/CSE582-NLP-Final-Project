from transformers import AutoTokenizer
import os
import pandas as pd
from ensemble_test import evaluate
from transformers import AutoModelForSeq2SeqLM
from memsum.summarizers import MemSum
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def abstractive(pathm,text):
  # generate a summary for the test dataset
  tokenizer = AutoTokenizer.from_pretrained(pathm,local_files_only=True, cache_dir="model/")
  inputs = tokenizer(text, return_tensors="pt").input_ids
  model = AutoModelForSeq2SeqLM.from_pretrained(pathm)
  outputs = model.generate(inputs, max_new_tokens=20, do_sample=False)
  final_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
  return final_text

def extractive(pathm,corpus):
  model = MemSum( pathm, "vocabulary_200dim.pkl", gpu = 0 ,  max_doc_len = 500  ) 
  for data in tqdm(corpus):
    extracted_summary = model.extract( [data["text"]], p_stop_thres = 0.6, max_extracted_sentences_per_document = 7 )[0]
    ext_text = "\n".join(extracted_summary)
  return ext_text


def main():
  #loading the dataset
  df = pd.read_csv('corpusfinal.csv')
  df_sample = df.sample(frac=0.5, replace=True, random_state=1)

  # #loading the model
  path_abstractive = "t5-small"
  path_extractive = "model_batch_3360.pt"

  # # generate a summary for the test dataset
  df_test = pd.DataFrame(columns=['gold_summary','modelsummary'])

  for index,row in df_sample.iterrows():

    text = row["document"]
    summary = row["abs_summary"]

    print("Original Text : ")
    print(text)
    print()

    # getting the extractive summary
    print("Extractive Summary : ")
    corpus_lst = []
    corpus_dict = {}
    corpus_dict['text'] = text.split(".")
    corpus_lst.append(corpus_dict)
    ext_text = extractive(path_extractive,corpus_lst)
    print(ext_text)
    print()

  #  # getting the abstractive summary
    print("Abstractive Summary : ")
    final_text = abstractive(path_abstractive,ext_text)
    print(final_text)

    print(25*"=")
    df_test = df_test.append({'gold_summary':summary,'modelsummary':final_text},ignore_index=True)

  scores = evaluate( df_test, 0.6, 7)
  print("ENSEMBLE ROGUE SCORES")
  print(scores)


if __name__ == "__main__":
    main()
    