import numpy as np
import evaluate
import nltk
from nltk.tokenize import sent_tokenize
from transformers import DataCollatorForSeq2Seq,Seq2SeqTrainer,Seq2SeqTrainingArguments,AutoModelForSeq2SeqLM,AutoModelForMaskedLM
import numpy as np
from helper import *


def train(args,dataset,model_checkpoint,tokenizer):
    rouge_score = evaluate.load("rouge")

    nltk.download("punkt")


    small_train,small_validation = generate_train_dataset(dataset,tokenizer,train_size = 1700, valid_size = 200)

    print(model_checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    # model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

    batch_size = args.batch_size
    num_train_epochs = args.epochs
    # Show the training loss with every epoch
    logging_steps = len(small_train) // batch_size
    model_name = model_checkpoint.split("/")[-1]

    args = Seq2SeqTrainingArguments(
        output_dir=f"{model_name}-finetuned-wildfire",
        evaluation_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=args.weight_decay,
        save_total_limit=3,
        num_train_epochs=num_train_epochs,
        predict_with_generate=True,
        logging_steps=logging_steps,
    )

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        # Decode generated summaries into text
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        # Decode reference summaries into text
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # ROUGE expects a newline after each sentence
        decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
        # Compute ROUGE scores
        result = rouge_score.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )
        # Extract the median scores
        result = {key: value*100 for key, value in result.items()}
        return {k: round(v, 4) for k, v in result.items()}

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=small_train,
        eval_dataset=small_validation,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    validation_metrics = trainer.evaluate()

    return trainer,validation_metrics

def test(trainer,test_data):
    test_obj = trainer.predict(test_data)
    return test_obj.metrics