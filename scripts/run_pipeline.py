"""
Complete pipeline runner for MovieLens-32M Generative Recommendation
"""
import os
import sys
import argparse
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from utils import setup_logging
from src.data_preprocessing import MovieLensPreprocessor
from src.train_rqvae import RQVAETrainer
from src.sequence_generator import SequenceGenerator
from src.train_tiger import TIGERTrainer
from src.evaluation import RecommendationEvaluator
from src.onerec_lite import OneRecLiteTrainer

def run_stage_0(config: Config):
    """Stage 0: Environment & Data Setup"""
    print("=" * 60)
    print("Stage 0: Environment & Data Setup")
    print("=" * 60)
    
    # Check if data exists
    data_dir = config.data.data_dir
    if not os.path.exists(os.path.join(data_dir, "ratings.csv")):
        print("ERROR: MovieLens-32M data not found!")
        print(f"Please download and extract ml-32m.zip to {data_dir}")
        print("Download from: https://files.grouplens.org/datasets/movielens/ml-32m.zip")
        return False
    
    print("✓ MovieLens-32M data found")
    print("✓ Environment setup complete")
    return True

def run_stage_1(config: Config):
    """Stage 1: Build Semantic IDs (RQ-VAE)"""
    print("=" * 60)
    print("Stage 1: Build Semantic IDs (RQ-VAE)")
    print("=" * 60)
    
    # Data preprocessing
    print("Step 1.1: Data Preprocessing...")
    preprocessor = MovieLensPreprocessor(
        data_dir=config.data.data_dir,
        min_rating=config.data.min_rating,
        min_interactions=config.data.min_interactions
    )
    
    ratings, movies, tfidf_features = preprocessor.process_data(config.output_dir)
    print(f"✓ Processed {len(ratings)} ratings and {len(movies)} movies")
    
    # Train RQ-VAE
    print("Step 1.2: Training RQ-VAE...")
    trainer = RQVAETrainer(config)
    model = trainer.train()
    print("✓ RQ-VAE training completed")
    
    # Generate semantic IDs
    print("Step 1.3: Generating Semantic IDs...")
    semantic_ids = trainer.generate_semantic_ids()
    print(f"✓ Generated semantic IDs for {len(semantic_ids)} items")
    
    return True

def run_stage_2(config: Config):
    """Stage 2: Generate Training Corpus"""
    print("=" * 60)
    print("Stage 2: Generate Training Corpus")
    print("=" * 60)
    
    generator = SequenceGenerator(
        min_rating=config.data.min_rating,
        max_seq_length=config.data.max_seq_length,
        min_interactions=config.data.min_interactions
    )
    
    sequences = generator.process_sequences(
        data_dir=config.output_dir,
        output_dir=os.path.join(config.output_dir, "sequences")
    )
    
    print(f"✓ Generated sequences for {len(sequences['train'])} training users")
    print(f"✓ Generated sequences for {len(sequences['val'])} validation users")
    print(f"✓ Generated sequences for {len(sequences['test'])} test users")
    
    return True

def run_stage_3(config: Config):
    """Stage 3: Train Lightweight TIGER"""
    print("=" * 60)
    print("Stage 3: Train Lightweight TIGER")
    print("=" * 60)
    
    trainer = TIGERTrainer(config)
    
    sequences_dir = os.path.join(config.output_dir, "sequences")
    model = trainer.train(sequences_dir, training_mode="seq2seq")
    
    print("✓ TIGER training completed")
    
    return True

def run_stage_4(config: Config):
    """Stage 4: Offline Evaluation"""
    print("=" * 60)
    print("Stage 4: Offline Evaluation")
    print("=" * 60)
    
    evaluator = RecommendationEvaluator(config)
    
    model_path = os.path.join(config.model_dir, "tiger_final")
    sequences_dir = os.path.join(config.output_dir, "sequences")
    
    results = evaluator.run_evaluation(model_path, sequences_dir, config.output_dir)
    
    # Print results
    print("\nEvaluation Results:")
    print("-" * 40)
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
    
    print("✓ Evaluation completed")
    
    return True

def run_stage_5(config: Config):
    """Stage 5: End-to-End OneRec-lite (Optional)"""
    print("=" * 60)
    print("Stage 5: End-to-End OneRec-lite (Optional)")
    print("=" * 60)
    
    trainer = OneRecLiteTrainer(config)
    
    base_model_path = os.path.join(config.model_dir, "tiger_final")
    sequences_dir = os.path.join(config.output_dir, "sequences")
    
    final_model = trainer.train_complete_pipeline(base_model_path, sequences_dir)
    
    print("✓ OneRec-lite training completed")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="MovieLens-32M Generative Recommendation Pipeline")
    parser.add_argument("--stages", type=str, default="0,1,2,3,4", 
                       help="Comma-separated list of stages to run (0,1,2,3,4,5)")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to custom config file")
    
    args = parser.parse_args()
    
    # Load config
    config = Config()
    
    # Setup logging
    logger = setup_logging(os.path.join(config.log_dir, "pipeline.log"))
    
    # Parse stages
    stages_to_run = [int(s.strip()) for s in args.stages.split(",")]
    
    print("MovieLens-32M Generative Recommendation Pipeline")
    print("=" * 60)
    print(f"Stages to run: {stages_to_run}")
    print(f"Output directory: {config.output_dir}")
    print(f"Model directory: {config.model_dir}")
    print("=" * 60)
    
    # Stage functions
    stage_functions = {
        0: run_stage_0,
        1: run_stage_1,
        2: run_stage_2,
        3: run_stage_3,
        4: run_stage_4,
        5: run_stage_5
    }
    
    # Run stages
    for stage in stages_to_run:
        if stage in stage_functions:
            try:
                success = stage_functions[stage](config)
                if not success:
                    print(f"❌ Stage {stage} failed!")
                    break
                print(f"✅ Stage {stage} completed successfully!")
            except Exception as e:
                print(f"❌ Stage {stage} failed with error: {e}")
                logger.error(f"Stage {stage} failed", exc_info=True)
                break
        else:
            print(f"❌ Unknown stage: {stage}")
            break
    
    print("\n" + "=" * 60)
    print("Pipeline execution completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
