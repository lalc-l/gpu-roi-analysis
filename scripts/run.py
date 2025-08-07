#!/usr/bin/env python3
"""GPU ROI Analysis Benchmark Suite.

This module provides comprehensive benchmarking functionality to compare
GPU performance and cost efficiency between H100 and B200 gpus.

Usage example:
    python roi_benchmark.py --gpu H100 --model meta-llama/Llama-3.1-8B-Instruct
    python roi_benchmark.py --test-mode --gpu H100
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pynvml as nvml
nvml.nvmlInit()  # Initialize NVML for GPU monitoring
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# GPU hourly pricing in USD
GPU_PRICING = {
    "H100": 2.69,
    "B200": 3.79
}

# Peak FLOPS estimates for MFU calculation (BF16 precision)
PEAK_FLOPS = {
    "H100": 2000e12,  # 2000 TFLOPS
    "B200": 4500e12   # 4500 TFLOPS
}

# Test model for validation mode
TEST_MODEL = "microsoft/DialoGPT-medium"

# Complete benchmarking suite of models
RECOMMENDED_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    "Qwen/Qwen3-32B",
    "meta-llama/LLama-4-Scout-17B-16E-Instruct",
    "deepseek/DeepSeek-V3-0324",
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
    "deepseek/DeepSeek-R1-0528"
]

class GPUMonitor:
    """Monitors GPU metrics during benchmark execution.
    
    This class provides real-time monitoring of GPU utilization, memory usage,
    and power consumption using NVIDIA Management Library (NVML).
    """
    
    def __init__(self):
        """Initializes GPU monitoring."""
        try:
            nvml.nvmlInit()
            self.device_count = nvml.nvmlDeviceGetCount()
            self.handles = [nvml.nvmlDeviceGetHandleByIndex(i) 
                          for i in range(self.device_count)]
            self.peak_memory_gb = 0.0
        except Exception as e:
            raise RuntimeError(f"Failed to initialize GPU monitoring: {e}")
    
    def get_current_metrics(self) -> List[Dict]:
        """Gets current GPU metrics for all devices.
        
        Returns:
            List of dictionaries containing GPU metrics for each device.
            Each dict contains: gpu_id, utilization_percent, memory_used_gb,
            memory_total_gb, memory_utilization_percent, power_watts.
        """
        metrics = []
        
        for i, handle in enumerate(self.handles):
            try:
                util = nvml.nvmlDeviceGetUtilizationRates(handle)
                mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                power = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                
                memory_used_gb = mem_info.used / (1024**3)
                memory_total_gb = mem_info.total / (1024**3)
                memory_util_percent = (mem_info.used / mem_info.total) * 100
                
                # Track peak memory usage
                self.peak_memory_gb = max(self.peak_memory_gb, memory_used_gb)
                
                metrics.append({
                    'gpu_id': i,
                    'utilization_percent': util.gpu,
                    'memory_used_gb': memory_used_gb,
                    'memory_total_gb': memory_total_gb,
                    'memory_utilization_percent': memory_util_percent,
                    'power_watts': power
                })
                
            except Exception as e:
                logging.warning(f"Failed to get metrics for GPU {i}: {e}")
                
        return metrics


class ROIBenchmark:
    """Main benchmark suite for GPU ROI analysis.
    
    This class orchestrates the complete benchmarking process including
    model loading, training simulation, inference testing, and financial
    analysis across different GPUs.
    """
    
    def __init__(self, gpu: str, model_name: str, test_mode: bool = False):
        """Initializes the ROI benchmark suite.
        
        Args:
            gpu: Target GPU ("H100" or "B200").
            model_name: HuggingFace model identifier to benchmark.
            test_mode: If True, runs quick validation with test model.
        """
        self.gpu = gpu
        self.model_name = TEST_MODEL if test_mode else model_name
        self.test_mode = test_mode
        self.monitor = GPUMonitor()
        self.model = None
        self.tokenizer = None
        self.model_params = 0
        
        # Initialize results structure
        self.results = {
            'metadata': {
                'gpu': gpu,
                'model_name': self.model_name,
                'test_mode': test_mode,
                'timestamp': datetime.now().isoformat(),
                'hostname': os.uname().nodename,
                'gpu_count': torch.cuda.device_count(),
                'cuda_version': torch.version.cuda,
                'pytorch_version': torch.__version__
            },
            'metrics': {}
        }
        
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Sets up logging configuration for benchmark execution."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Log file naming convention: H100_model_test_01-05_001
        month_day = datetime.now().strftime("%m-%d")
        model_safe = self.model_name.replace('/', '_').replace('-', '_')
        
        # Add timestamp to log file name
        timestamp = datetime.now().strftime("%H%M%S")  # HHMMSS format
        mode_suffix = "_test" if self.test_mode else ""
        log_file = log_dir / f"{self.gpu}_{model_safe}{mode_suffix}_{month_day}_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        mode_str = "TEST MODE" if self.test_mode else "PRODUCTION"
        logging.info(f"Starting benchmark [{mode_str}]: {self.model_name} on {self.gpu}")
    
    def load_model(self) -> bool:
        """Loads the specified model and tokenizer."""
        
        logging.info(f"Loading model: {self.model_name}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"
            
            # Load model with single GPU placement for fair comparison
            device_map = "cuda:0" if torch.cuda.device_count() > 1 else "auto"
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
                trust_remote_code=True
            )
            
            # Calculate model parameters
            self.model_params = sum(p.numel() for p in self.model.parameters())
            
            logging.info(f"Model loaded successfully: {self.model_params:,} parameters")
            logging.info(f"Model device: {next(self.model.parameters()).device}")
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            return False
    
    def _calculate_mfu(self, tokens_per_second: float) -> float:
        """Calculates Model FLOPs Utilization (MFU).
        
        Args:
            tokens_per_second: Measured throughput in tokens per second.
            
        Returns:
            MFU percentage (0-100).
        """
        # Estimate ~6 FLOPs per parameter per token for transformer inference
        model_flops_per_token = 6 * self.model_params
        actual_flops_per_second = model_flops_per_token * tokens_per_second
        
        # Updated peak FLOPS for better accuracy
        peak_flops_h100 = 1000e12  # 1000 TFLOPS for bfloat16
        peak_flops_b200 = 2500e12  # Estimated 2500 TFLOPS for bfloat16
        
        peak_flops = peak_flops_b200 if self.gpu == "B200" else peak_flops_h100
        mfu = (actual_flops_per_second / peak_flops) * 100
        
        return min(mfu, 100.0)  # Cap at 100%
    
    def _calculate_perplexity(self, texts: List[str], sample_size: int = 10) -> Optional[float]:
        """Calculates perplexity on sample texts for accuracy validation.
        
        Args:
            texts: List of text samples to evaluate.
            sample_size: Number of samples to use for perplexity calculation.
        """
        if not texts:
            return None
            
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        try:
            with torch.no_grad():
                for text in texts[:sample_size]:
                    inputs = self.tokenizer(
                        text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512
                    )
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                    
                    outputs = self.model(**inputs, labels=inputs["input_ids"])
                    loss = outputs.loss
                    
                    total_loss += loss.item() * inputs["input_ids"].size(1)
                    total_tokens += inputs["input_ids"].size(1)
            
            if total_tokens == 0:
                return None
                
            avg_loss = total_loss / total_tokens
            perplexity = math.exp(avg_loss)
            
            return perplexity
            
        except Exception as e:
            logging.warning(f"Perplexity calculation failed: {e}")
            return None
    
    def _get_training_dataset(self) -> List[str]:
        """Generates training dataset for benchmark.
        
        Returns:
            List of training text samples.
        """
        if self.test_mode:
            # Simple dataset for test mode
            return [
                "The future of artificial intelligence will",
                "Machine learning algorithms can process",
                "Deep neural networks are designed to",
                "Natural language processing enables"
            ] * 25  # 100 samples for quick test
        else:
            # More comprehensive dataset for production benchmarks
            return [
                "The future of artificial intelligence will transform industries by enabling automated decision-making and intelligent data analysis across multiple domains.",
                "Machine learning algorithms process vast amounts of data to identify complex patterns and make predictions that would be impossible for humans to detect manually.",
                "Deep neural networks use multiple layers of interconnected neurons to learn hierarchical representations of data, enabling breakthrough performance in computer vision and natural language processing.",
                "Natural language processing allows computers to understand, interpret, and generate human language, facilitating more natural human-machine interactions.",
                "Computer vision systems analyze visual information to recognize objects, detect anomalies, and extract meaningful insights from images and video streams.",
                "Reinforcement learning agents learn optimal strategies through trial and error interactions with their environment, maximizing cumulative rewards over time.",
                "Graph neural networks process structured data to understand complex relationships between entities, enabling applications in social networks, molecular analysis, and knowledge graphs.",
                "Transformer architectures have revolutionized natural language understanding by using self-attention mechanisms to capture long-range dependencies in sequential data.",
                "Large language models demonstrate emergent capabilities in reasoning, code generation, and creative writing through training on diverse text corpora at massive scale.",
                "Multimodal AI systems integrate information from multiple data types including text, images, and audio to provide more comprehensive understanding and generation capabilities."
            ] * 40  # 400 samples for thorough benchmarking
    
    def run_training_benchmark(self) -> None:
        """Executes comprehensive training benchmark.
        
        Measures training performance including throughput, memory usage,
        loss convergence, and calculates financial metrics.
        """
        # Add optimizer for training
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)



        logging.info("Starting training benchmark")
        self.model.train()
        
        # Get training dataset
        training_texts = self._get_training_dataset()
        
         # Configure training parameters
        if self.test_mode:
            num_batches = 10
            batch_size = 32
            max_length = 2048
        else:
            num_batches = 25  
            batch_size = 64
            max_length = 4096

        # Pre-tokenize training data with consistent batch sizes
        logging.info("Pre-tokenizing training data with optimized batches")
        all_batches = []

        # Create consistent batches that match your batch_size
        for i in range(0, len(training_texts), batch_size):
            batch_texts = training_texts[i:i+batch_size]
            # Ensure batch is exactly batch_size by repeating texts if needed
            while len(batch_texts) < batch_size:
                batch_texts.append(training_texts[0])
            
            inputs = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=max_length,  # Use the max_length variable you defined
                return_tensors="pt"
            )
            
            all_batches.append({
                'input_ids': inputs.input_ids.to(self.model.device),
                'attention_mask': inputs.attention_mask.to(self.model.device)
            })

        logging.info(f"Created {len(all_batches)} batches of size {batch_size}")

        # Initialize tracking variables
        total_tokens = 0
        losses = []
        batch_times = []
        self.monitor.peak_memory_gb = 0.0
        
        logging.info(f"Running {num_batches} training batches")
        
        # Training loop
        start_time = time.time()
        
        for i in tqdm(range(num_batches), desc="Training batches"):
            batch_start = time.time()
            
            # Get pre-tokenized batch data
            batch_data = all_batches[i % len(all_batches)]
            batch_input_ids = batch_data['input_ids']
            batch_attention = batch_data['attention_mask']
            
            # Forward pass
            with torch.amp.autocast('cuda',dtype=torch.bfloat16):
                outputs = self.model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention,
                    labels=batch_input_ids
                )
                loss = outputs.loss
            
            # Backward pass with gradient accumulation
            accumulation_steps = 4
            loss = loss / accumulation_steps  # Scale loss
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            # Track metrics
            batch_end = time.time()
            batch_time = batch_end - batch_start
            batch_times.append(batch_time)
            
            batch_tokens = batch_input_ids.numel()
            total_tokens += batch_tokens
            losses.append(loss.item())
            
            if (i + 1) % 5 == 0:
                logging.info(f"Batch {i+1}/{num_batches} - Loss: {loss.item():.4f}")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate performance metrics
        tokens_per_second = total_tokens / total_time
        mfu = self._calculate_mfu(tokens_per_second)
        
        # Get current GPU metrics
        gpu_metrics = self.monitor.get_current_metrics()[0]  # Primary GPU
        
        # Calculate financial metrics
        gpu_hour_cost = GPU_PRICING[self.gpu]
        hours_for_1m_tokens = (1_000_000 / tokens_per_second) / 3600
        cost_per_million_tokens = hours_for_1m_tokens * gpu_hour_cost
        performance_per_dollar = tokens_per_second / gpu_hour_cost
        
        # Calculate accuracy metrics
        loss_convergence = losses[0] - losses[-1] if len(losses) > 1 else 0.0
        perplexity = self._calculate_perplexity(training_texts[:20])
        
        # Calculate time to target accuracy (loss < 2.5 as example target)
        target_loss = 2.5
        time_to_target = None
        cost_to_target = None
        
        for i, loss_val in enumerate(losses):
            if loss_val < target_loss:
                time_to_target = (i + 1) * (total_time / len(losses))
                cost_to_target = (time_to_target / 3600) * gpu_hour_cost
                break
        
        # Store training results
        training_results = {
            'total_training_time_seconds': total_time,
            'total_tokens_processed': total_tokens,
            'tokens_per_second': tokens_per_second,
            'mfu_percent': mfu,
            'memory_usage_gb': gpu_metrics['memory_used_gb'],
            'memory_utilization_percent': gpu_metrics['memory_utilization_percent'],
            'peak_memory_usage_gb': self.monitor.peak_memory_gb,
            'final_loss': losses[-1],
            'loss_convergence': loss_convergence,
            'perplexity_score': perplexity,
            'cost_per_million_tokens': cost_per_million_tokens,
            'performance_per_dollar': performance_per_dollar,
            'average_batch_time_seconds': sum(batch_times) / len(batch_times),
            'time_to_target_accuracy_seconds': time_to_target,
            'cost_to_target_accuracy': cost_to_target,
            'target_loss_threshold': target_loss
        }
        
        self.results['metrics']['training'] = training_results
        
        logging.info(f"Training completed - Tokens/sec: {tokens_per_second:.1f}, MFU: {mfu:.1f}%")
    
    def run_inference_benchmark(self) -> None:
        """Executes comprehensive inference benchmark.
        
        Tests inference performance across multiple batch sizes,
        measuring throughput, latency, and cost efficiency.
        """
        logging.info("Starting inference benchmark")
        self.model.eval()
        
        # Test prompts for inference
        test_prompts = [
            "The development of artificial intelligence requires",
            "Machine learning models achieve high performance through",
            "Deep learning architectures are designed to",
            "Natural language understanding involves analyzing",
            "Computer vision systems can process",
            "Reinforcement learning enables agents to"
        ]
        
        # Test different batch sizes
        batch_sizes = [1, 4] if self.test_mode else [1, 4, 8]
        inference_results = {}
        
        for batch_size in batch_sizes:
            logging.info(f"Testing inference batch size: {batch_size}")
            
            # Prepare batch prompts
            if batch_size > len(test_prompts):
                batch_prompts = test_prompts * ((batch_size // len(test_prompts)) + 1)
                batch_prompts = batch_prompts[:batch_size]
            else:
                batch_prompts = test_prompts[:batch_size]
            
            throughputs = []
            latencies = []
            
            # Run inference iterations
            num_iterations = 5 if self.test_mode else 10
            
            for iteration in range(num_iterations):
                inputs = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1024
                ).to(self.model.device)
                
                # Warmup iteration
                if iteration == 0:
                    with torch.no_grad():
                        _ = self.model.generate(
                            inputs.input_ids,
                            max_new_tokens=10,
                            do_sample=False,
                            pad_token_id=self.tokenizer.eos_token_id
                        )
                    torch.cuda.synchronize()
                    continue
                
                # Benchmark iteration
                torch.cuda.synchronize()
                start_time = time.time()
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs.input_ids,
                        max_new_tokens=64,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                
                torch.cuda.synchronize()
                end_time = time.time()
                
                # Calculate metrics
                inference_time = end_time - start_time
                tokens_generated = batch_size * 64
                throughput = tokens_generated / inference_time
                latency_per_token = (inference_time * 1000) / tokens_generated
                
                throughputs.append(throughput)
                latencies.append(latency_per_token)
            
            # Calculate batch statistics
            avg_throughput = sum(throughputs) / len(throughputs)
            avg_latency = sum(latencies) / len(latencies)
            
            # Calculate financial metrics
            gpu_hour_cost = GPU_PRICING[self.gpu]
            tokens_per_hour = avg_throughput * 3600
            cost_per_million_tokens = (gpu_hour_cost * 1_000_000) / tokens_per_hour
            performance_per_dollar = avg_throughput / gpu_hour_cost
            
            # Store batch results
            inference_results[f'batch_{batch_size}'] = {
                'average_throughput_tokens_per_sec': avg_throughput,
                'average_latency_ms_per_token': avg_latency,
                'cost_per_million_tokens': cost_per_million_tokens,
                'performance_per_dollar': performance_per_dollar,
                'iterations_tested': len(throughputs)
            }
            
            logging.info(f"Batch {batch_size} - Throughput: {avg_throughput:.1f} tok/s, "
                        f"Latency: {avg_latency:.2f} ms/tok")
        
        self.results['metrics']['inference'] = inference_results
    
    def calculate_financial_projections(self) -> None:
        """Calculates multi-year cost projections for different usage scenarios."""
        if 'training' not in self.results['metrics']:
            return
        
        training_metrics = self.results['metrics']['training']
        gpu_hour_cost = GPU_PRICING[self.gpu]
        tokens_per_hour = training_metrics['tokens_per_second'] * 3600
        
        # Define usage scenarios (hours per year)
        scenarios = {
            'light_usage': 8 * 22 * 12,      # 8 hrs/day, 22 business days/month
            'medium_usage': 16 * 30 * 12,    # 16 hrs/day, 30 days/month  
            'heavy_usage': 24 * 30 * 12      # 24/7 operation
        }
        
        projections = {}
        for scenario, hours_per_year in scenarios.items():
            yearly_cost = hours_per_year * gpu_hour_cost
            tokens_per_year = hours_per_year * tokens_per_hour
            
            projections[scenario] = {
                'year_1_cost_usd': yearly_cost,
                'year_2_cost_usd': yearly_cost * 2,
                'year_3_cost_usd': yearly_cost * 3,
                'tokens_per_year': tokens_per_year,
                'total_hours_per_year': hours_per_year
            }
        
        self.results['metrics']['financial_projections'] = projections
    
    def save_results(self) -> Tuple[str, str]:
        """Saves benchmark results to JSON and CSV files.
        
        Returns:
            Tuple of (json_file_path, csv_file_path).
        """
        results_dir = Path("results/raw")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        month_day = datetime.now().strftime("%m-%d")
        model_safe = self.model_name.replace('/', '_').replace('-', '_')
        mode_suffix = "_test" if self.test_mode else ""

        # Save results
        timestamp = datetime.now().strftime("%H%M%S")
        mode_suffix = "_test" if self.test_mode else ""
        json_file = results_dir / f"{self.gpu}_{model_safe}{mode_suffix}_{month_day}_{timestamp}.json"
        csv_file = results_dir / f"{self.gpu}_{model_safe}{mode_suffix}_{month_day}_{timestamp}.csv"
        
        
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save CSV summary
        csv_file = results_dir / f"{self.gpu}_{model_safe}{mode_suffix}_{month_day}_{timestamp}.csv"
        self._save_csv_summary(csv_file)
        
        logging.info(f"Results saved to: {json_file}")
        logging.info(f"CSV summary saved to: {csv_file}")
        
        return str(json_file), str(csv_file)
    
    def _save_csv_summary(self, csv_file: Path) -> None:
        """Saves CSV summary with all key metrics for spreadsheet analysis.
        
        Args:
            csv_file: Path where CSV file should be saved.
        """
        training_metrics = self.results['metrics'].get('training', {})
        inference_metrics = self.results['metrics'].get('inference', {})
        
        # Use batch_4 as representative inference metrics
        repr_inference = inference_metrics.get('batch_4', 
                                              inference_metrics.get('batch_1', {}))
        
        # Create comprehensive summary row
        summary_data = {
            'GPU': self.gpu,
            'Model_Name': self.model_name,
            'Test_Mode': self.test_mode,
            'Timestamp': self.results['metadata']['timestamp'],
            'GPU_Count': self.results['metadata']['gpu_count'],
            
            # Memory metrics
            'Memory_Usage_GB': training_metrics.get('memory_usage_gb', 0),
            'Memory_Utilization_Percent': training_metrics.get('memory_utilization_percent', 0),
            'Peak_Memory_Usage_GB': training_metrics.get('peak_memory_usage_gb', 0),
            
            # Performance metrics
            'Training_Tokens_Per_Sec_GPU': training_metrics.get('tokens_per_second', 0),
            'Training_MFU_Percent': training_metrics.get('mfu_percent', 0),
            'Inference_Tokens_Per_Sec_GPU': repr_inference.get('average_throughput_tokens_per_sec', 0),
            'Inference_Latency_Ms_Per_Token': repr_inference.get('average_latency_ms_per_token', 0),
            
            # Cost and ROI metrics
            'Training_Cost_Per_Million_Tokens': training_metrics.get('cost_per_million_tokens', 0),
            'Training_Performance_Per_Dollar': training_metrics.get('performance_per_dollar', 0),
            'Inference_Cost_Per_Million_Tokens': repr_inference.get('cost_per_million_tokens', 0),
            'Inference_Performance_Per_Dollar': repr_inference.get('performance_per_dollar', 0),
            
            # Accuracy validation metrics
            'Final_Loss': training_metrics.get('final_loss', 0),
            'Loss_Convergence': training_metrics.get('loss_convergence', 0),
            'Perplexity_Score': training_metrics.get('perplexity_score', 0),
            'Time_To_Target_Accuracy_Seconds': training_metrics.get('time_to_target_accuracy_seconds', 0),
            'Cost_To_Target_Accuracy': training_metrics.get('cost_to_target_accuracy', 0),
        }
        
        # Add financial projections if available
        projections = self.results['metrics'].get('financial_projections', {})
        if projections:
            for scenario in ['light_usage', 'medium_usage', 'heavy_usage']:
                if scenario in projections:
                    summary_data.update({
                        f'{scenario}_year_1_cost': projections[scenario]['year_1_cost_usd'],
                        f'{scenario}_year_2_cost': projections[scenario]['year_2_cost_usd'],
                        f'{scenario}_year_3_cost': projections[scenario]['year_3_cost_usd'],
                    })
        
        df = pd.DataFrame([summary_data])
        df.to_csv(csv_file, index=False)


def main():
    """Main entry point for the ROI benchmark suite."""
    parser = argparse.ArgumentParser(
        description="GPU ROI Analysis Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python roi_benchmark.py --gpu H100 --model meta-llama/Llama-3.1-8B-Instruct
  python roi_benchmark.py --test-mode --gpu H100
  python roi_benchmark.py --gpu B200 --model meta-llama/Llama-3.1-70B-Instruct --skip-inference
        """
    )
    
    parser.add_argument(
        "--gpu", 
        required=True, 
        choices=["H100", "B200"], 
        help="Target GPU for benchmarking"
    )
    parser.add_argument(
        "--model", 
        help="HuggingFace model identifier (ignored in test mode)"
    )
    parser.add_argument(
        "--test-mode", 
        action="store_true", 
        help="Run quick validation test with DialoGPT-medium"
    )
    parser.add_argument(
        "--skip-training", 
        action="store_true", 
        help="Skip training benchmark"
    )
    parser.add_argument(
        "--skip-inference", 
        action="store_true", 
        help="Skip inference benchmark"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.test_mode and not args.model:
        parser.error("--model is required unless using --test-mode")
    
    if args.skip_training and args.skip_inference:
        parser.error("Cannot skip both training and inference benchmarks")
    
    # Initialize and run benchmark
    benchmark = ROIBenchmark(args.gpu, args.model, args.test_mode)
    
    if not benchmark.load_model():
        logging.error("Failed to load model. Exiting.")
        sys.exit(1)
    
    try:
        # Run benchmarks based on configuration
        if not args.skip_training:
            benchmark.run_training_benchmark()
        
        if not args.skip_inference:
            benchmark.run_inference_benchmark()
        
        # Calculate financial projections
        benchmark.calculate_financial_projections()
        
        # Save results
        json_file, csv_file = benchmark.save_results()
        
        logging.info("ROI benchmark completed successfully")
        logging.info(f"Detailed results: {json_file}")
        logging.info(f"Summary for spreadsheet: {csv_file}")
        
    except Exception as e:
        logging.error(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()