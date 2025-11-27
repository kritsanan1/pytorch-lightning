#!/usr/bin/env python3
"""
Phin AI Training Script
Fine-tune a language model on Thai xylophone (phin) music data for business applications.
"""

import os
import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling,
    pipeline
)
from datasets import Dataset, DatasetDict
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PhinDataProcessor:
    """Process and prepare phin music data for training."""
    
    def __init__(self, data_dir: str = "phin_ai_training_project"):
        self.data_dir = Path(data_dir)
        self.processed_data = []
        
    def load_and_process_data(self) -> List[Dict]:
        """Load and process all available phin data."""
        logger.info("Loading phin music data...")
        
        # Load the documentation and research papers
        docs_path = self.data_dir / "04_documentation" / "phin_resources_summary.md"
        final_summary = self.data_dir / "FINAL_SUMMARY.md"
        
        processed_data = []
        
        # Process documentation files
        if docs_path.exists():
            with open(docs_path, 'r', encoding='utf-8') as f:
                content = f.read()
                processed_data.append({
                    "text": content,
                    "source": "phin_resources_summary",
                    "type": "documentation"
                })
        
        if final_summary.exists():
            with open(final_summary, 'r', encoding='utf-8') as f:
                content = f.read()
                processed_data.append({
                    "text": content,
                    "source": "final_summary",
                    "type": "summary"
                })
        
        # Load YouTube video list for training context
        youtube_path = self.data_dir / "05_youtube_links" / "youtube_video_list.md"
        if youtube_path.exists():
            with open(youtube_path, 'r', encoding='utf-8') as f:
                content = f.read()
                processed_data.append({
                    "text": content,
                    "source": "youtube_videos",
                    "type": "video_list"
                })
        
        # Load research paper
        research_path = self.data_dir / "03_research_papers" / "KMUTT_Thai_Xylophone_Transcription_2024.pdf"
        if research_path.exists():
            # For PDF, we'll create a placeholder - in real implementation you'd extract text
            processed_data.append({
                "text": "KMUTT Thai Xylophone Transcription Research Paper 2024 - Automatic Music Transcription for Thai Xylophone with 98.54% accuracy using EWMA method.",
                "source": "research_paper",
                "type": "research"
            })
        
        # Generate training data for Thai music patterns
        thai_music_patterns = self._generate_thai_music_patterns()
        processed_data.extend(thai_music_patterns)
        
        logger.info(f"Processed {len(processed_data)} documents")
        return processed_data
    
    def _generate_thai_music_patterns(self) -> List[Dict]:
        """Generate training data for Thai music patterns and terminology."""
        patterns = [
            {
                "text": "ลายนกไส่บินข้ามทุ่ง เป็นลายพิณอีสานที่มีลักษณะจังหวะเร็วและมีลูกเล่นมาก ใช้สำหรับการแสดงในงานต่างๆ ของภาคอีสาน",
                "source": "thai_patterns",
                "type": "music_theory"
            },
            {
                "text": "ระบบโน๊ตดนตรีไทยใช้ 7 เสียงหลัก ได้แก่ ด ร ม ฟ ซ ล ท ซึ่งแตกต่างจากระบบสากลที่ใช้ 12 เสียง การแปลงเสียงพิณเป็นโน๊ตต้องคำนึงถึงระบบ 7 เสียงนี้",
                "source": "thai_patterns", 
                "type": "music_theory"
            },
            {
                "text": "พิณอีสานมีลายต่างๆ เช่น ลายลำเพลิน ลายแมลงภู่ตอมดอกไม้ ลายเต้ยโขง ลายเต้ยพม่า ลายโปงลาง ลายเซิ้งบั้งไฟ ลายลำเต้ย ลายศรีโคตรบูรณ์ แต่ละลายมีจังหวะและลักษณะเฉพาะ",
                "source": "thai_patterns",
                "type": "music_theory"
            },
            {
                "text": "Spotify Basic Pitch เป็นเครื่องมือสำหรับแปลงเสียงเป็น MIDI ที่เหมาะสำหรับการเริ่มต้น แต่ต้องปรับแต่งสำหรับดนตรีไทยเพราะรูปแบบเสียงและระบบโน๊ตแตกต่างจากดนตรีสากล",
                "source": "thai_patterns",
                "type": "ai_methodology"
            },
            {
                "text": "การ transcribe เสียงพิณต้องพิจารณาหลายปัจจัย ได้แก่ การตรวจจับ onset (จังหวะเริ่มต้น) การตรวจจับ pitch (ระดับเสียง) และการจัดการกับลูกเล่นพิเศษของพิณอีสาน",
                "source": "thai_patterns",
                "type": "ai_methodology"
            }
        ]
        return patterns

class PhinTrainer:
    """Trainer for fine-tuning language models on phin data."""
    
    def __init__(self, 
                 model_name: str = "microsoft/DialoGPT-medium",
                 output_dir: str = "phin_model_output",
                 max_length: int = 512):
        self.model_name = model_name
        self.output_dir = output_dir
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        
    def setup_model(self):
        """Set up the model and tokenizer."""
        logger.info(f"Loading model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        logger.info("Model loaded successfully")
        
    def prepare_dataset(self, data: List[Dict]) -> Dataset:
        """Prepare dataset for training."""
        logger.info("Preparing dataset...")
        
        # Combine all text data
        all_texts = []
        for item in data:
            all_texts.append(item["text"])
        
        # Create dataset
        dataset_dict = {
            "text": all_texts
        }
        
        dataset = Dataset.from_dict(dataset_dict)
        
        # Tokenize the dataset
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"]
        )
        
        logger.info(f"Dataset prepared with {len(tokenized_dataset)} examples")
        return tokenized_dataset
    
    def train(self, dataset: Dataset, epochs: int = 3, batch_size: int = 4):
        """Train the model."""
        logger.info("Starting training...")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            save_steps=100,
            save_total_limit=2,
            logging_steps=10,
            learning_rate=5e-5,
            weight_decay=0.01,
            warmup_steps=100,
            prediction_loss_only=True,
            remove_unused_columns=False,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset,
        )
        
        # Train the model
        trainer.train()
        
        # Save the model
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        logger.info("Training completed!")
        
    def create_inference_pipeline(self):
        """Create inference pipeline for the trained model."""
        return pipeline(
            "text-generation",
            model=self.output_dir,
            tokenizer=self.output_dir,
            device=0 if torch.cuda.is_available() else -1
        )

def main():
    """Main training function."""
    logger.info("Starting Phin AI Training Process")
    
    # Initialize data processor
    data_processor = PhinDataProcessor()
    
    # Load and process data
    training_data = data_processor.load_and_process_data()
    
    if not training_data:
        logger.error("No training data found. Please check your data directory.")
        return
    
    # Initialize trainer
    trainer = PhinTrainer(
        model_name="microsoft/DialoGPT-medium",
        output_dir="phin_model_output",
        max_length=512
    )
    
    # Setup model
    trainer.setup_model()
    
    # Prepare dataset
    dataset = trainer.prepare_dataset(training_data)
    
    # Train the model
    trainer.train(dataset, epochs=5, batch_size=2)
    
    # Create inference pipeline
    pipe = trainer.create_inference_pipeline()
    
    # Test the model
    logger.info("Testing the trained model...")
    test_prompts = [
        "ลายพิณอีสานที่นิยมที่สุดคือ",
        "การแปลงเสียงพิณเป็นโน๊ตดนตรีควรใช้",
        "Spotify Basic Pitch เหมาะสำหรับ"
    ]
    
    for prompt in test_prompts:
        logger.info(f"\nPrompt: {prompt}")
        result = pipe(prompt, max_length=100, num_return_sequences=1)
        logger.info(f"Generated: {result[0]['generated_text']}")
    
    logger.info("Phin AI Training completed successfully!")

if __name__ == "__main__":
    main()