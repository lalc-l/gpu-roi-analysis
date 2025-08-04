"""
GPU ROI Analysis - Benchmark Runner
Professional benchmarking suite for H100 vs B200 comparison
"""

import os
import sys
import json
import time
import logging
import argparse
import pandas as pd
from datetime import datetime
from pathlib import Path

import torch
import nvidia_ml_py3 as nvml
from transformers import AutoTokenizer, AutoModelForCausalLM

# Setup logging
def setup_logging(platform, model_name):
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"benchmark_{platform}_{model_name.replace('/', '_')}_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return str(log_file)

class GPUMonitor:
    """GPU performance monitoring utility"""
    
    def __init__(self):
        nvml.nvmlInit()
        self.device_count = nvml.nvmlDeviceGetCount()
        self.handles = [nvml.nvmlDeviceGetHandleByIndex(i) for i in range(self.device_count)]
    
    def get_metrics(self):
        """Get current GPU metrics"""
        metrics = []
        for i, handle in enumerate(self.handles):
            try:
                util = nvml.nvmlDeviceGetUtilizationRates(handle)
                mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                power = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                
                metrics.append({
                    'gpu_id': i,
                    'utilization_percent': util.gpu,
                    'memory_used_mb': mem_info.used // (1024 * 1024),
                    'memory_total_mb': mem_info.total // (1024 * 1024),
                    'memory_utilization_percent': (mem_info.used / mem_info.total) * 100,
                    'power_watts': power,
                    'timestamp': time.time()
                })
            except Exception as e:
                logging.warning(f"Failed to get metrics for GPU {i}: {e}")
        
        return metrics

class BenchmarkRunner:
    """Main benchmark execution class"""
    
    def __init__(self, platform, model_name, test_type):
        self.platform = platform
        self.model_name = model_name  
        self.test_type = test_type
        self.monitor = GPUMonitor()
        self.results = {
            'metadata': {
                'platform': platform,
                'model_name': model_name,
                'test_type': test_type,
                'timestamp': datetime.now().isoformat(),
                'hostname': os.uname().nodename,
                'gpu_count': torch.cuda.device_count(),
                'cuda_version': torch.version.cuda,
                'pytorch_version': torch.__version__
            },
            'performance': {},
            'system_metrics': []
        }
    
    def load_model(self):
        """Load model and tokenizer"""
        logging.info(f"Loading model: {self.model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Force single GPU for fair comparison
            if torch.cuda.device_count() > 1:
                logging.info(f"Multi-GPU system detected. Using GPU 0 for single-GPU comparison.")
                device_map = {"": 0}
            else:
                device_map = "auto"
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
                trust_remote_code=True
            )
            
            param_count = sum(p.numel() for p in self.model.parameters())
            logging.info(f"Model loaded successfully. Parameters: {param_count:,}")
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            return False
    
    def run_training_benchmark(self):
        """Execute training benchmark"""
        logging.info("Starting training benchmark")
        
        # Training dataset
        texts = [
            "The future of artificial intelligence will transform industries through",
            "Machine learning algorithms enable computers to learn patterns from",
            "Deep neural networks process complex data by using multiple layers of",
            "Natural language processing allows machines to understand human communication via",
            "Computer vision systems can analyze and interpret visual information using",
            "Reinforcement learning agents optimize their behavior through trial and error in"
        ] * 100  # 600 samples
        
        self.model.train()
        
        # Tokenize dataset
        logging.info("Tokenizing training data")
        inputs = self.tokenizer(
            texts, 
            truncation=True, 
            padding=True, 
            max_length=512, 
            return_tensors="pt"
        )
        
        input_ids = inputs.input_ids.to(self.model.device)
        attention_mask = inputs.attention_mask.to(self.model.device)
        
        # Training loop
        num_batches = 20
        batch_size = len(texts) // num_batches
        total_tokens = 0
        batch_times = []
        
        logging.info(f"Running {num_batches} training batches")
        
        start_time = time.time()
        
        for i in range(num_batches):
            batch_start = time.time()
            
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(texts))
            
            batch_input_ids = input_ids[start_idx:end_idx]
            batch_attention = attention_mask[start_idx:end_idx]
            
            # Forward pass
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = self.model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention,
                    labels=batch_input_ids
                )
                loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            batch_end = time.time()
            batch_time = batch_end - batch_start
            batch_times.append(batch_time)
            
            batch_tokens = batch_input_ids.numel()
            total_tokens += batch_tokens
            
            if (i + 1) % 5 == 0:
                logging.info(f"Batch {i+1}/{num_batches} - Loss: {loss.item():.4f} - Time: {batch_time:.2f}s")
        
        end_time = time.time()
        total_time = end_time - start_time
        tokens_per_second = total_tokens / total_time
        
        self.results['performance'].update({
            'total_training_time_seconds': total_time,
            'total_tokens_processed': total_tokens,
            'tokens_per_second': tokens_per_second,
            'average_batch_time_seconds': sum(batch_times) / len(batch_times),
            'final_loss': loss.item(),
            'batches_completed': num_batches
        })
        
        logging.info(f"Training completed - Tokens/sec: {tokens_per_second:.1f}")
        
    def run_inference_benchmark(self):
        """Execute inference benchmark"""
        logging.info("Starting inference benchmark")
        
        self.model.eval()
        
        test_prompts = [
            "The development of artificial intelligence systems requires",
            "Machine learning models achieve high performance through",
            "Deep learning architectures are designed to process",
            "Natural language understanding involves analyzing"
        ]
        
        batch_sizes = [1, 4, 8, 16]
        inference_results = {}
        
        for batch_size in batch_sizes:
            if batch_size > len(test_prompts):
                batch_prompts = test_prompts * ((batch_size // len(test_prompts)) + 1)
                batch_prompts = batch_prompts[:batch_size]
            else:
                batch_prompts = test_prompts[:batch_size]
            
            logging.info(f"Testing inference batch size: {batch_size}")
            
            throughputs = []
            latencies = []
            
            # Run multiple iterations
            for iteration in range(10):
                inputs = self.tokenizer(
                    batch_prompts, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True
                ).to(self.model.device)
                
                # Warmup on first iteration
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
                
                inference_time = end_time - start_time
                tokens_generated = batch_size * 64
                throughput = tokens_generated / inference_time
                latency_per_token = (inference_time * 1000) / tokens_generated  # ms per token
                
                throughputs.append(throughput)
                latencies.append(latency_per_token)
            
            # Calculate statistics
            avg_throughput = sum(throughputs) / len(throughputs)
            avg_latency = sum(latencies) / len(latencies)
            
            inference_results[f'batch_{batch_size}'] = {
                'average_throughput_tokens_per_sec': avg_throughput,
                'average_latency_ms_per_token': avg_latency,
                'iterations': len(throughputs)
            }
            
            logging.info(f"Batch {batch_size} - Throughput: {avg_throughput:.1f} tok/s, Latency: {avg_latency:.2f} ms/tok")
        
        self.results['performance']['inference'] = inference_results
        
    def save_results(self):
        """Save benchmark results"""
        results_dir = Path("results/raw")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON
        json_file = results_dir / f"{self.platform}_{self.model_name.replace('/', '_')}_{self.test_type}_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save CSV summary
        csv_file = results_dir / f"{self.platform}_{self.model_name.replace('/', '_')}_{self.test_type}_{timestamp}.csv"
        
        if self.test_type == "training":
            data = [{
                'platform': self.platform,
                'model': self.model_name,
                'test_type': self.test_type,
                'timestamp': self.results['metadata']['timestamp'],
                'tokens_per_second': self.results['performance']['tokens_per_second'],
                'total_time_seconds': self.results['performance']['total_training_time_seconds'],
                'total_tokens': self.results['performance']['total_tokens_processed'],
                'final_loss': self.results['performance']['final_loss']
            }]
        else:  # inference
            data = []
            for batch_key, batch_results in self.results['performance']['inference'].items():
                data.append({
                    'platform': self.platform,
                    'model': self.model_name,
                    'test_type': self.test_type,
                    'batch_size': batch_key.split('_')[1],
                    'timestamp': self.results['metadata']['timestamp'],
                    'throughput_tokens_per_sec': batch_results['average_throughput_tokens_per_sec'],
                    'latency_ms_per_token': batch_results['average_latency_ms_per_token']
                })
        
        df = pd.DataFrame(data)
        df.to_csv(csv_file, index=False)
        
        logging.info(f"Results saved to: {json_file}")
        logging.info(f"CSV summary saved to: {csv_file}")
        
        return str(json_file), str(csv_file)

def main():
    parser = argparse.ArgumentParser(description="GPU ROI Analysis Benchmark Runner")
    parser.add_argument("--platform", required=True, choices=["H100", "B200"], help="GPU platform")
    parser.add_argument("--model", required=True, help="Model name (e.g., microsoft/DialoGPT-medium)")
    parser.add_argument("--test-type", required=True, choices=["training", "inference"], help="Test type")
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = setup_logging(args.platform, args.model)
    logging.info(f"Starting benchmark: {args.platform} - {args.model} - {args.test_type}")
    logging.info(f"Log file: {log_file}")
    
    # Initialize benchmark
    runner = BenchmarkRunner(args.platform, args.model, args.test_type)
    
    # Load model
    if not runner.load_model():
        logging.error("Failed to load model. Exiting.")
        sys.exit(1)
    
    # Run benchmark
    try:
        if args.test_type == "training":
            runner.run_training_benchmark()
        else:
            runner.run_inference_benchmark()
        
        # Save results
        json_file, csv_file = runner.save_results()
        
        logging.info("Benchmark completed successfully")
        logging.info(f"Results: {json_file}")
        logging.info(f"Summary: {csv_file}")
        
    except Exception as e:
        logging.error(f"Benchmark failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
