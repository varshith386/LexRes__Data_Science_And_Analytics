import os
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset

# Genrates Facts
def read_dataset(reference_dir, summaries_dir):
    dataset = []
    for file in os.listdir(reference_dir):
        if file.endswith(".txt"):
            file_id = file[:-4]  # Assuming the file extension is .txt and getting the base filename
            with open(os.path.join(reference_dir, file), 'r') as f:
                main_text = f.read()

            facts_summary = open(os.path.join(summaries_dir, "Facts", f"{file_id}_facts.txt"), 'r').read()

            dataset.append({
                "main_text": main_text,
                "facts_summary": facts_summary,
            })
    return dataset

# Load dataset
reference_dir = r'dataset\IN-Ext\judgement'  # Change this to your reference files path
summaries_dir = r'FIRAC SUMMARIES'  # Change this to your summaries files path
dataset = read_dataset(reference_dir, summaries_dir)

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
        facts_summary = data['facts_summary']
        input_texts.append(f"summarize: {main_text[:512]} [facts]")
        target_texts.append(facts_summary)
    return input_texts, target_texts

train_input_texts, train_target_texts = prepare_training_data(train_dataset)
eval_input_texts, eval_target_texts = prepare_training_data(eval_dataset)

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

# Save the model
model.save_pretrained("./trained_model")
tokenizer.save_pretrained("./trained_model")

# Function to generate facts summary
def generate_facts(model, tokenizer, text):
    input_text = f"summarize: {text[:512]} [facts]"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output_ids = model.generate(input_ids, max_new_tokens=150)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Example usage
def summarize_facts(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
    return generate_facts(model, tokenizer, text)

example_file_path = r'dataset\IN-Ext\judgement\1953_L_1.txt'  # Change this to the path of your example text file
facts_summary = summarize_facts(example_file_path)
print(facts_summary)
