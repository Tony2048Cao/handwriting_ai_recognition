import torch
from torch.utils.data import Dataset, DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainingArguments, Seq2SeqTrainer, default_data_collator
from PIL import Image
import pymongo
from typing import List, Dict
import matplotlib.pyplot as plt
import os
import numpy as np

# MongoDB connection
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client.ai_training
collection = db.get_collection("handwriting_training_data")

# Paths for model and output
TROCR_BASE_HANDWRITTEN = 'handwriting_ai_recognition/source/modules/trocr_base_handwritten'
TROCR_FINE_TUNING_HANDWRITTEN = "handwriting_ai_recognition/source/modules/trocr_finetuned_01"
LOSS_PLOT_PATH = "handwriting_ai_recognition/source/modules/trocr_finetuned_01/training_results/training_losses.png"

# Ensure the output directory exists
os.makedirs(os.path.dirname(LOSS_PLOT_PATH), exist_ok=True)

# Check for available GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the dataset
class HandwritingDataset(Dataset):
    def __init__(self, processor):
        self.processor = processor
        self.data = list(collection.find({}))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item['image_path']).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()
        labels = self.processor.tokenizer(item['text'], 
                                          padding="max_length", 
                                          max_length=512).input_ids
        
        return {"pixel_values": pixel_values, "labels": torch.tensor(labels)}

# Custom trainer to track losses
class CustomTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_losses = []
        self.eval_losses = []

    def training_step(self, model, inputs):
        loss = super().training_step(model, inputs)
        self.train_losses.append(loss.item())
        return loss

    def evaluation_loop(self, *args, **kwargs):
        output = super().evaluation_loop(*args, **kwargs)
        self.eval_losses.append(output.metrics['eval_loss'])
        return output

    def plot_losses(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Training Loss')
        
        # Use only common data points
        common_length = min(len(self.train_losses), len(self.eval_losses))
        eval_indices = np.linspace(0, len(self.train_losses) - 1, common_length)
        plt.plot(eval_indices, self.eval_losses[:common_length], label='Validation Loss')
        
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.savefig(LOSS_PLOT_PATH)
        plt.close()
        print(f"Loss plot saved to {LOSS_PLOT_PATH}")

def handwriting_module_finetuning():
    # Load pre-trained model and processor
    model_name = TROCR_BASE_HANDWRITTEN
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)

    # Set decoder_start_token_id
    model.config.decoder_start_token_id = 50256  # Or another suitable value for your model

    # Move model to GPU
    model = model.to(device)

    # Load model and tokenizer
    tokenizer = processor.tokenizer
    
    # Ensure tokenizer has pad_token, if not, set one
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Assign tokenizer's pad_token_id to model config
    model.config.pad_token_id = tokenizer.pad_token_id

    # Prepare dataset
    dataset = HandwritingDataset(processor)
    train_size = int(0.9 * len(dataset))
    train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])

    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=TROCR_FINE_TUNING_HANDWRITTEN,
        evaluation_strategy="epoch",
        learning_rate=5e-4,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        num_train_epochs=50,
        weight_decay=0.01,
        save_total_limit=3,
        fp16=True,  # Enable mixed precision training
        dataloader_num_workers=4,  # Increase number of worker processes for data loading
        save_steps=10,  # Save checkpoint every 10 steps
        save_strategy="steps",  # Save based on steps
        gradient_accumulation_steps=2,  # Gradient accumulation steps
    )

    # Define Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor.tokenizer,
        data_collator=default_data_collator,
    )

    # Start training
    trainer.train()

    # Save fine-tuned model
    trainer.save_model(TROCR_FINE_TUNING_HANDWRITTEN)

    # Plot and save loss curve
    trainer.plot_losses()

    print(f"Training completed. Loss curve saved to {LOSS_PLOT_PATH}")

def evaluate_handwriting_model():
    # Load fine-tuned model and processor
    model_name = TROCR_FINE_TUNING_HANDWRITTEN
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)

    # Move model to GPU
    model = model.to(device)

    # Prepare test dataset
    test_dataset = HandwritingDataset(processor)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Set model to evaluation mode
    model.eval()
    
    total_cer = 0
    total_samples = 0

    with torch.no_grad():
        for batch in test_loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            # Generate predictions
            outputs = model.generate(pixel_values)

            # Decode predictions and labels
            pred_texts = processor.batch_decode(outputs, skip_special_tokens=True)
            label_texts = processor.batch_decode(labels, skip_special_tokens=True)

            # Calculate Character Error Rate (CER)
            for pred, label in zip(pred_texts, label_texts):
                cer = calculate_cer(pred, label)
                total_cer += cer
                total_samples += 1

    average_cer = total_cer / total_samples
    print(f"Average Character Error Rate: {average_cer:.4f}")

def calculate_cer(pred: str, label: str) -> float:
    # Calculate Character Error Rate using edit distance
    edit_distance = levenshtein_distance(pred, label)
    return edit_distance / len(label)

def levenshtein_distance(s1: str, s2: str) -> int:
    # Calculate Levenshtein distance between two strings
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

if __name__ == "__main__":
    print(f"Using device: {device}")
    print('-'*10 + ' Fine-tuning ' + '-'*10)
    handwriting_module_finetuning()

    print('-'*10 + ' Evaluating ' + '-'*10)
    evaluate_handwriting_model()

