
from summarizers import MemSum
from rouge_score import rouge_scorer
import numpy as np
import pandas as pd

def evaluate( corpus, p_stop, max_extracted_sentences ):
    scores = []
    rouge_cal = rouge_scorer.RougeScorer(['rouge1','rouge2', 'rougeLsum'], use_stemmer=False)
    for index,rows in corpus.iterrows():
        gold_summary = rows["abs_summary"]
        extracted_summary = rows["modelsummary"]
        # extracted_summary = model.extract( [data["text"]], p_stop_thres = p_stop, max_extracted_sentences_per_document = max_extracted_sentences )[0]
        # score = rouge_cal.score( "\n".join( gold_summary ), "\n".join(extracted_summary)  )
        score = rouge_cal.score( gold_summary, extracted_summary  )
        scores.append( [score["rouge1"].fmeasure, score["rouge2"].fmeasure, score["rougeLsum"].fmeasure ] )
    
    return np.asarray(scores).mean(axis = 0)

def main():
    df = pd.read_csv('data/corpusfinal.csv')
    df_test = df.iloc[:int(len(df)*0.3)]
    evaluate( df_test, 0.6, 7)

if __name__ == "__main__":
    main()