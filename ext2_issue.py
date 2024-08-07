import os
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset

# Genrates Issues
def read_dataset(main_text_dir, issues_dir):
    dataset = []
    for file in os.listdir(issues_dir):
        if file.endswith("_issues.txt"):
            file_id = file[:-11]  # Removing the '_issues.txt' suffix to get the base filename
            main_text_file = f"{file_id}.txt"
            issues_summary_file = file  # This is the same as the current file

            main_text_path = os.path.join(main_text_dir, main_text_file)
            issues_summary_path = os.path.join(issues_dir, issues_summary_file)

            print(f"Trying to match main text file: {main_text_path} with issues summary file: {issues_summary_path}")

            if os.path.exists(main_text_path):  # Ensure the main text file exists
                with open(main_text_path, 'r') as f:
                    main_text = f.read()

                with open(issues_summary_path, 'r') as f:
                    issues_summary = f.read()

                dataset.append({
                    "main_text": main_text,
                    "issues_summary": issues_summary,
                })
            else:
                print(f"Main text file does not exist: {main_text_path}")
    return dataset

# Load dataset
main_text_dir = r'dataset\IN-Ext\judgement'  # Change this to your main texts directory path
issues_dir = r'FIRAC SUMMARIES\ISSUES'  # Change this to your issues directory path
dataset = read_dataset(main_text_dir, issues_dir)

# Check if dataset is loaded correctly
print(f"Total examples loaded: {len(dataset)}")
if len(dataset) == 0:
    print("No examples found. Please check the directory paths and file naming conventions.")
    exit()

# Split the dataset into training and evaluation sets
train_size = int(0.8 * len(dataset))
train_dataset = dataset[:train_size]
eval_dataset = dataset[train_size:]

# Prepare data for training
def prepare_training_data(dataset):
    input_texts = []
    target_texts = []
    for data in dataset:
        main_text = data['main_text']
        issues_summary = data['issues_summary']
        input_texts.append(main_text)
        target_texts.append(f"summarize: {issues_summary} [issue]")
    return input_texts, target_texts

train_input_texts, train_target_texts = prepare_training_data(train_dataset)
eval_input_texts, eval_target_texts = prepare_training_data(eval_dataset)

# Check if data preparation is done correctly
print(f"Training examples: {len(train_input_texts)}")
print(f"Evaluation examples: {len(eval_input_texts)}")
if len(train_input_texts) == 0 or len(eval_input_texts) == 0:
    print("Training or evaluation data is empty. Please check the data preparation steps.")
    exit()

# Create Hugging Face datasets
train_hf_dataset = Dataset.from_dict({"input_texts": train_input_texts, "target_texts": train_target_texts})
eval_hf_dataset = Dataset.from_dict({"input_texts": eval_input_texts, "target_texts": eval_target_texts})

# Load pre-trained T5 tokenizer and model
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Tokenize data
def tokenize_function(examples):
    model_inputs = tokenizer(examples["input_texts"], max_length=512, truncation=True)
    labels = tokenizer(examples["target_texts"], max_length=150, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train_dataset = train_hf_dataset.map(tokenize_function, batched=True)
tokenized_eval_dataset = eval_hf_dataset.map(tokenize_function, batched=True)

# Check if tokenization is done correctly
print(f"Tokenized training dataset size: {len(tokenized_train_dataset)}")
print(f"Tokenized evaluation dataset size: {len(tokenized_eval_dataset)}")
if len(tokenized_train_dataset) == 0 or len(tokenized_eval_dataset) == 0:
    print("Tokenized training or evaluation dataset is empty. Please check the tokenization steps.")
    exit()

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=100,
    evaluation_strategy="epoch",
    eval_steps=50,
)

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer
)

# Train the model
trainer.train()

# Function to generate issues summary
def generate_issue(model, tokenizer, text):
    input_text = text
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    output_ids = model.generate(input_ids, max_length=150, num_beams=2, length_penalty=0.5, early_stopping=True)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Example usage
def summarize_issues(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
    return generate_issue(model, tokenizer, text)

example_file_path = r'dataset\IN-Ext\judgement\1953_L_1.txt'  # Change this to the path of your example text file
issues_summary = summarize_issues(example_file_path)
print(issues_summary)
