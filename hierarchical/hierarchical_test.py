
from rouge_score import rouge_scorer
import numpy as np
import pandas as pd

def evaluate( corpus, p_stop, max_extracted_sentences ):
    scores = []
    rouge_cal = rouge_scorer.RougeScorer(['rouge1','rouge2', 'rougeLsum'], use_stemmer=False)
    for index,rows in corpus.iterrows():
        gold_summary = rows["gold_summary"]
        extracted_summary = rows["modelsummary"]
        score = rouge_cal.score( gold_summary, extracted_summary  )
        scores.append( [score["rouge1"].fmeasure, score["rouge2"].fmeasure, score["rougeLsum"].fmeasure ] )
    
    return np.asarray(scores).mean(axis = 0)

def main():
    df = pd.read_csv('../dataset/corpusfinal.csv')
    df_test = df.iloc[:int(len(df)*0.1)]
    print(evaluate( df_test, 0.6, 7))

if __name__ == "__main__":
    main()