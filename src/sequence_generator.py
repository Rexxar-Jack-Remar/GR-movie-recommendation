"""
Generate training sequences for generative recommendation
"""
import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Tuple
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class SequenceGenerator:
    """Generate training sequences from user interactions"""
    
    def __init__(self, min_rating: float = 4.0, max_seq_length: int = 50, 
                 min_interactions: int = 5):
        self.min_rating = min_rating
        self.max_seq_length = max_seq_length
        self.min_interactions = min_interactions
        
    def load_data(self, data_dir: str) -> Tuple[pd.DataFrame, Dict[int, List[int]]]:
        """Load processed data and semantic IDs"""
        # Load processed ratings
        ratings_path = os.path.join(data_dir, "processed_ratings.csv")
        ratings = pd.read_csv(ratings_path)
        
        # Load semantic IDs
        semantic_ids_path = os.path.join(data_dir, "item_semantic_ids.jsonl")
        semantic_ids = {}
        with open(semantic_ids_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                semantic_ids[item['movieId']] = item['semantic_ids']
        
        return ratings, semantic_ids
    
    def create_user_sequences(self, ratings: pd.DataFrame) -> Dict[int, List[int]]:
        """Create user interaction sequences"""
        logger.info("Creating user sequences...")
        
        # Filter positive ratings
        positive_ratings = ratings[ratings['rating'] >= self.min_rating].copy()
        
        # Sort by timestamp
        positive_ratings = positive_ratings.sort_values(['userId', 'timestamp'])
        
        # Group by user and create sequences
        user_sequences = {}
        for user_id, group in positive_ratings.groupby('userId'):
            sequence = group['movieId'].tolist()
            
            # Filter users with minimum interactions
            if len(sequence) < self.min_interactions:
                continue
                
            # Keep only the most recent interactions
            if len(sequence) > self.max_seq_length:
                sequence = sequence[-self.max_seq_length:]
                
            user_sequences[user_id] = sequence
        
        logger.info(f"Created sequences for {len(user_sequences)} users")
        return user_sequences
    
    def convert_to_semantic_sequences(self, user_sequences: Dict[int, List[int]], 
                                    semantic_ids: Dict[int, List[int]]) -> Dict[int, List[str]]:
        """Convert movie IDs to semantic ID sequences"""
        logger.info("Converting to semantic sequences...")
        
        semantic_sequences = {}
        for user_id, movie_sequence in user_sequences.items():
            semantic_sequence = []
            for movie_id in movie_sequence:
                if movie_id in semantic_ids:
                    # Convert semantic IDs to string tokens
                    semantic_tokens = [f"<id_{sid}>" for sid in semantic_ids[movie_id]]
                    semantic_sequence.extend(semantic_tokens)
                else:
                    # Use unknown token if movie not found
                    semantic_sequence.extend(["<unk>", "<unk>"])
            
            semantic_sequences[user_id] = semantic_sequence
        
        return semantic_sequences
    
    def split_sequences(self, user_sequences: Dict[int, List], 
                       test_ratio: float = 0.2, val_ratio: float = 0.1) -> Tuple[Dict, Dict, Dict]:
        """Split sequences into train/val/test"""
        logger.info("Splitting sequences...")
        
        train_seqs, val_seqs, test_seqs = {}, {}, {}
        
        for user_id, sequence in user_sequences.items():
            if len(sequence) < 6:  # Need at least 6 tokens for meaningful split
                continue
                
            seq_len = len(sequence)
            test_size = max(2, int(seq_len * test_ratio))  # At least 2 tokens for test
            val_size = max(2, int(seq_len * val_ratio))    # At least 2 tokens for val
            train_size = seq_len - test_size - val_size
            
            if train_size < 2:  # Need at least 2 tokens for training
                continue
                
            train_seqs[user_id] = sequence[:train_size]
            val_seqs[user_id] = sequence[train_size:train_size + val_size]
            test_seqs[user_id] = sequence[train_size + val_size:]
        
        logger.info(f"Split: {len(train_seqs)} train, {len(val_seqs)} val, {len(test_seqs)} test users")
        return train_seqs, val_seqs, test_seqs
    
    def generate_training_samples(self, sequences: Dict[int, List[str]]) -> List[Dict]:
        """Generate training samples in next-item prediction format"""
        logger.info("Generating training samples...")
        
        samples = []
        for user_id, sequence in sequences.items():
            # Generate samples with sliding window
            for i in range(1, len(sequence)):
                input_seq = sequence[:i]
                target = sequence[i]
                
                samples.append({
                    'user_id': user_id,
                    'input_sequence': ' '.join(input_seq),
                    'target': target,
                    'input_length': len(input_seq)
                })
        
        logger.info(f"Generated {len(samples)} training samples")
        return samples
    
    def generate_text_format(self, sequences: Dict[int, List[str]], 
                           output_path: str, format_type: str = "causal"):
        """Generate text format for language model training"""
        logger.info(f"Generating {format_type} format...")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for user_id, sequence in sequences.items():
                if format_type == "causal":
                    # Causal language modeling format
                    text = f"<user_{user_id}> " + " ".join(sequence) + " <eos>"
                    f.write(text + '\n')
                elif format_type == "seq2seq":
                    # Sequence-to-sequence format
                    for i in range(1, len(sequence)):
                        input_seq = " ".join(sequence[:i])
                        target = sequence[i]
                        f.write(f"{input_seq}\t{target}\n")
        
        logger.info(f"Saved {format_type} format to {output_path}")
    
    def process_sequences(self, data_dir: str, output_dir: str):
        """Complete sequence processing pipeline"""
        logger.info("Starting sequence processing...")
        
        # Load data
        ratings, semantic_ids = self.load_data(data_dir)
        
        # Create user sequences
        user_sequences = self.create_user_sequences(ratings)
        
        # Convert to semantic sequences
        semantic_sequences = self.convert_to_semantic_sequences(user_sequences, semantic_ids)
        
        # Split sequences
        train_seqs, val_seqs, test_seqs = self.split_sequences(semantic_sequences)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate different formats
        formats = {
            'train': train_seqs,
            'val': val_seqs,
            'test': test_seqs
        }
        
        for split, sequences in formats.items():
            # Causal LM format
            causal_path = os.path.join(output_dir, f"{split}_causal.txt")
            self.generate_text_format(sequences, causal_path, "causal")
            
            # Seq2seq format
            seq2seq_path = os.path.join(output_dir, f"{split}_seq2seq.txt")
            self.generate_text_format(sequences, seq2seq_path, "seq2seq")
            
            # JSON format for detailed analysis
            json_path = os.path.join(output_dir, f"{split}_sequences.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(sequences, f, ensure_ascii=False, indent=2)
        
        # Generate training samples
        train_samples = self.generate_training_samples(train_seqs)
        samples_path = os.path.join(output_dir, "train_samples.json")
        with open(samples_path, 'w', encoding='utf-8') as f:
            json.dump(train_samples, f, ensure_ascii=False, indent=2)
        
        logger.info("Sequence processing completed!")
        
        return {
            'train': train_seqs,
            'val': val_seqs,
            'test': test_seqs,
            'samples': train_samples
        }

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from config import Config
    from utils import setup_logging
    
    config = Config()
    logger = setup_logging(os.path.join(config.log_dir, "sequence_generation.log"))
    
    generator = SequenceGenerator(
        min_rating=config.data.min_rating,
        max_seq_length=config.data.max_seq_length,
        min_interactions=config.data.min_interactions
    )
    
    sequences = generator.process_sequences(
        data_dir=config.output_dir,
        output_dir=os.path.join(config.output_dir, "sequences")
    )
    
    print(f"Generated sequences for {len(sequences['train'])} training users")
