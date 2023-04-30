from datasets import load_dataset
from transformers import AutoTokenizer
import argparse
from helper import *
import os
from model import *
import pandas as pd

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default = 4)
    parser.add_argument('--epochs', type=int,default = 5)
    parser.add_argument('--lr', type=float,default=5.6e-5)
    parser.add_argument('--weight_decay', type=float,default=0.001)
    parser.add_argument('--model', type=str,default = "t5-small")
    parser.add_argument('--output_path', type=str,default = "results")

    return parser.parse_args()

def main():
    args = get_args()
    df = pd.read_csv('corpusfinal.csv')
    model_checkpoint = args.model
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    dataset = convert_pd_to_gig(df)
    # training the model
    trainer,val_results = train(args,dataset,model_checkpoint,tokenizer)
    print("Testing Wildfire dataset")
    print(val_results)
    print(25*"=")
    trainer.save_model(model_checkpoint)

    # testing the model
    test_data = generate_test_dataset(dataset,tokenizer)
    test_results = test(trainer,test_data)
    print("Testing Wildfire dataset")
    print(test_results)
    print(25*"=")

    # generate a summary for the test dataset
    # print("Testing Wildfire dataset")
    # tokenizer = AutoTokenizer.from_pretrained("model/",local_files_only=True, cache_dir="model/")
    # text = ""
    # inputs = tokenizer(text, return_tensors="pt").input_ids
    # model = AutoModelForSeq2SeqLM.from_pretrained("model/")
    # outputs = model.generate(inputs, max_new_tokens=15, do_sample=False)
    # print(outputs)
    # print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    
    isExist = os.path.exists(str(args.output_path)+"_"+model_checkpoint)
    if not isExist:
        os.makedirs(str(args.output_path)+"_"+model_checkpoint)

    write2file(str(args.output_path)+"_"+model_checkpoint,val_results,dataset="wildfire",dataset_type="training")
    write2file(str(args.output_path)+"_"+model_checkpoint,test_results,dataset="wildfire",dataset_type="testing")

if __name__ == "__main__":
    main()