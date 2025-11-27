#!/usr/bin/env python3
"""
Phin AI Training Data Preprocessor

This module handles preprocessing of Thai phin (xylophone) music data for LLM training.
It processes audio files, transcripts, and metadata to create training datasets.
"""

import os
import json
import logging
import tarfile
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import librosa
import soundfile as sf
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PhinDataPreprocessor:
    """
    Preprocessor for Thai phin music data specifically designed for LLM training.
    Processes audio, transcripts, and metadata to create training datasets.
    """
    
    def __init__(self, 
                 data_dir: str = "/home/user/webapp/phin_ai_training_project",
                 output_dir: str = "/home/user/webapp/processed_data",
                 sample_rate: int = 22050,
                 max_audio_length: float = 30.0,
                 overlap: float = 0.5):
        """
        Initialize the preprocessor.
        
        Args:
            data_dir: Directory containing phin training data
            output_dir: Directory for processed data
            sample_rate: Audio sample rate
            max_audio_length: Maximum audio length in seconds
            overlap: Overlap ratio for audio segmentation
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.sample_rate = sample_rate
        self.max_audio_length = max_audio_length
        self.overlap = overlap
        
        # Create output directories
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "audio_segments").mkdir(exist_ok=True)
        (self.output_dir / "transcripts").mkdir(exist_ok=True)
        (self.output_dir / "metadata").mkdir(exist_ok=True)
        
        logger.info(f"Initialized PhinDataPreprocessor with data_dir: {self.data_dir}")
    
    def extract_audio_from_videos(self, video_list_path: str) -> List[str]:
        """
        Extract audio from YouTube videos using yt-dlp.
        
        Args:
            video_list_path: Path to file containing YouTube video URLs
            
        Returns:
            List of extracted audio file paths
        """
        audio_files = []
        
        try:
            with open(video_list_path, 'r') as f:
                urls = [line.strip() for line in f if line.strip()]
            
            for i, url in enumerate(urls):
                logger.info(f"Processing video {i+1}/{len(urls)}: {url}")
                
                # Extract audio using yt-dlp
                output_filename = f"phin_audio_{i+1:03d}.wav"
                output_path = self.output_dir / "audio_segments" / output_filename
                
                cmd = [
                    "yt-dlp",
                    "-f", "bestaudio",
                    "--extract-audio",
                    "--audio-format", "wav",
                    "--audio-quality", "0",
                    "-o", str(output_path),
                    url
                ]
                
                try:
                    subprocess.run(cmd, check=True, capture_output=True)
                    audio_files.append(str(output_path))
                    logger.info(f"Successfully extracted audio: {output_filename}")
                except subprocess.CalledProcessError as e:
                    logger.error(f"Failed to extract audio from {url}: {e}")
                    continue
                    
        except FileNotFoundError:
            logger.error(f"Video list file not found: {video_list_path}")
        except Exception as e:
            logger.error(f"Error extracting audio: {e}")
            
        return audio_files
    
    def segment_audio(self, audio_path: str, segment_length: float = 30.0) -> List[str]:
        """
        Segment long audio files into smaller chunks.
        
        Args:
            audio_path: Path to audio file
            segment_length: Length of each segment in seconds
            
        Returns:
            List of segmented audio file paths
        """
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            total_length = len(audio) / sr
            
            # Calculate segments
            segments = []
            step_size = int(segment_length * sr * (1 - self.overlap))
            segment_samples = int(segment_length * sr)
            
            for i in range(0, len(audio) - segment_samples + 1, step_size):
                start_sample = i
                end_sample = min(i + segment_samples, len(audio))
                
                if end_sample - start_sample >= segment_samples * 0.8:  # At least 80% of segment length
                    segment_audio = audio[start_sample:end_sample]
                    
                    # Create segment filename
                    base_name = Path(audio_path).stem
                    segment_filename = f"{base_name}_seg_{len(segments):03d}.wav"
                    segment_path = self.output_dir / "audio_segments" / segment_filename
                    
                    # Save segment
                    sf.write(segment_path, segment_audio, self.sample_rate)
                    segments.append(str(segment_path))
            
            logger.info(f"Segmented {audio_path} into {len(segments)} segments")
            return segments
            
        except Exception as e:
            logger.error(f"Error segmenting audio {audio_path}: {e}")
            return []
    
    def generate_training_text(self, audio_path: str) -> str:
        """
        Generate training text for phin music based on audio characteristics.
        This creates descriptive text that an LLM can learn from.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Generated training text
        """
        try:
            # Load audio and extract features
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Extract musical features
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0]
            
            # Determine phin style based on features
            avg_chroma = np.mean(chroma, axis=1)
            dominant_notes = np.argsort(avg_chroma)[-3:][::-1]
            
            # Map to Thai musical notes
            thai_notes = ['ซ', 'ร', 'ม', 'ฟ', 'ซ', 'ล', 'ท']
            dominant_thai_notes = [thai_notes[i] for i in dominant_notes if i < len(thai_notes)]
            
            # Generate descriptive text
            training_text = f"""This is a Thai phin (อีสาน xylophone) music recording.

Musical Characteristics:
- Tempo: {tempo:.1f} BPM
- Duration: {len(audio)/sr:.1f} seconds
- Sample Rate: {sr} Hz
- Dominant Notes: {', '.join(dominant_thai_notes)}
- Spectral Centroid: {np.mean(spectral_centroids):.0f} Hz
- Zero Crossing Rate: {np.mean(zero_crossing_rate):.3f}

Cultural Context:
This recording represents traditional Isan (Northeastern Thailand) music featuring the phin, a traditional Thai xylophone. The phin is an essential instrument in Isan culture, used in various ceremonial and entertainment contexts.

Musical Analysis:
The audio shows characteristics typical of {self._classify_phin_style(avg_chroma)} style phin music. The dominant notes suggest a {self._analyze_mode(avg_chroma)} mode, common in traditional Isan compositions.

Technical Notes:
- Audio quality: Suitable for machine learning analysis
- Background noise: {self._assess_noise_level(zero_crossing_rate)}
- Dynamic range: {self._assess_dynamic_range(audio)}
- Recommended for: Music transcription, cultural preservation, AI training
"""
            
            return training_text
            
        except Exception as e:
            logger.error(f"Error generating training text for {audio_path}: {e}")
            return f"Thai phin music recording: {Path(audio_path).name}"
    
    def _classify_phin_style(self, chroma_features: np.ndarray) -> str:
        """Classify phin playing style based on chroma features."""
        # Simplified classification based on note patterns
        if np.std(chroma_features) > 0.3:
            return "dynamic and expressive"
        elif np.mean(chroma_features[:3]) > np.mean(chroma_features[3:]):
            return "traditional folk"
        else:
            return "contemporary"
    
    def _analyze_mode(self, chroma_features: np.ndarray) -> str:
        """Analyze the musical mode."""
        # Simplified mode analysis
        peak_notes = np.argsort(chroma_features)[-2:][::-1]
        if abs(peak_notes[0] - peak_notes[1]) in [2, 4]:
            return "pentatonic"
        else:
            return "heptatonic"
    
    def _assess_noise_level(self, zcr: np.ndarray) -> str:
        """Assess background noise level."""
        zcr_mean = np.mean(zcr)
        if zcr_mean < 0.05:
            return "Low"
        elif zcr_mean < 0.1:
            return "Moderate"
        else:
            return "High"
    
    def _assess_dynamic_range(self, audio: np.ndarray) -> str:
        """Assess dynamic range."""
        dynamic_range = 20 * np.log10(np.max(np.abs(audio))) - 20 * np.log10(np.mean(np.abs(audio)))
        if dynamic_range > 20:
            return "Wide"
        elif dynamic_range > 10:
            return "Moderate"
        else:
            return "Narrow"
    
    def create_dataset(self, audio_files: List[str], output_format: str = "json") -> str:
        """
        Create a training dataset from audio files.
        
        Args:
            audio_files: List of audio file paths
            output_format: Output format ("json" or "huggingface")
            
        Returns:
            Path to created dataset file
        """
        dataset = []
        
        for i, audio_path in enumerate(audio_files):
            logger.info(f"Processing file {i+1}/{len(audio_files)}: {audio_path}")
            
            # Generate training text
            training_text = self.generate_training_text(audio_path)
            
            # Create dataset entry
            entry = {
                "audio_path": audio_path,
                "text": training_text,
                "filename": Path(audio_path).name,
                "duration": librosa.get_duration(path=audio_path),
                "sample_rate": self.sample_rate
            }
            
            dataset.append(entry)
        
        # Save dataset
        if output_format == "json":
            output_path = self.output_dir / "phin_training_dataset.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
        else:  # huggingface format
            output_path = self.output_dir / "phin_training_dataset"
            hf_dataset = Dataset.from_list(dataset)
            hf_dataset.save_to_disk(str(output_path))
        
        logger.info(f"Created dataset with {len(dataset)} entries: {output_path}")
        return str(output_path)
    
    def prepare_for_litgpt(self, dataset_path: str, tokenizer_name: str = "microsoft/DialoGPT-medium") -> str:
        """
        Prepare dataset for LitGPT training.
        
        Args:
            dataset_path: Path to dataset file
            tokenizer_name: Name of tokenizer to use
            
        Returns:
            Path to processed dataset for LitGPT
        """
        # Load dataset
        if dataset_path.endswith('.json'):
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            dataset = load_dataset(str(dataset_path))
            data = dataset['train']
        
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Process for LitGPT format
        litgpt_data = []
        for entry in data:
            text = entry['text']
            
            # Tokenize
            tokens = tokenizer.encode(text, max_length=1024, truncation=True)
            
            litgpt_entry = {
                "text": text,
                "tokens": tokens,
                "metadata": {
                    "audio_path": entry.get("audio_path", ""),
                    "filename": entry.get("filename", ""),
                    "duration": entry.get("duration", 0)
                }
            }
            
            litgpt_data.append(litgpt_entry)
        
        # Save in LitGPT format
        output_path = self.output_dir / "litgpt_phin_dataset.jsonl"
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in litgpt_data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        logger.info(f"Created LitGPT dataset: {output_path}")
        return str(output_path)


def main():
    """Main function to demonstrate the preprocessing pipeline."""
    
    # Initialize preprocessor
    preprocessor = PhinDataPreprocessor()
    
    # Example: Process existing audio files (if available)
    audio_dir = Path("/home/user/webapp/phin_ai_training_project/01_raw_audio_sources")
    if audio_dir.exists():
        audio_files = list(audio_dir.glob("*.wav"))
        logger.info(f"Found {len(audio_files)} audio files")
        
        if audio_files:
            # Create dataset
            dataset_path = preprocessor.create_dataset(audio_files)
            logger.info(f"Dataset created: {dataset_path}")
            
            # Prepare for LitGPT
            litgpt_path = preprocessor.prepare_for_litgpt(dataset_path)
            logger.info(f"LitGPT dataset ready: {litgpt_path}")
        else:
            logger.info("No audio files found. Please download videos first.")
    else:
        logger.info(f"Audio directory not found: {audio_dir}")
        logger.info("Please download videos using the provided scripts first.")


if __name__ == "__main__":
    main()