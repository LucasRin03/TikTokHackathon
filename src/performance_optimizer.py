#!/usr/bin/env python3
"""
NLP Pipeline Performance Optimizer
Analyzes and optimizes performance for large-scale review processing
"""

import pandas as pd
import numpy as np
import time
import psutil
import gc
from memory_profiler import profile
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import sys
from pathlib import Path

# Import NLP components
from advanced_nlp_features import AdvancedNLPFeatureExtractor
from policy_detection_system import PolicyViolationDetector
from similarity_analysis import ReviewSimilarityAnalyzer
from topic_modeling import RestaurantTopicModeler
from keyword_extraction import RestaurantKeywordExtractor

class NLPPerformanceOptimizer:
    """
    Comprehensive performance analysis and optimization for NLP pipeline
    Includes memory profiling, speed optimization, and scalability analysis
    """
    
    def __init__(self):
        print("⚡ Initializing NLP Performance Optimizer...")
        
        # Performance metrics
        self.metrics = {
            'processing_times': {},
            'memory_usage': {},
            'throughput': {},
            'bottlenecks': [],
            'optimization_suggestions': []
        }
        
        # System information
        self.system_info = {
            'cpu_count': mp.cpu_count(),
            'memory_total': psutil.virtual_memory().total / (1024**3),  # GB
            'memory_available': psutil.virtual_memory().available / (1024**3)  # GB
        }
        
        print(f"✅ System: {self.system_info['cpu_count']} CPUs, {self.system_info['memory_total']:.1f}GB RAM")
    
    def benchmark_component_performance(self, df_sample, component_name, component_func, *args):
        """Benchmark individual component performance"""
        
        print(f"🔧 Benchmarking {component_name}...")
        
        # Memory before
        process = psutil.Process()
        memory_before = process.memory_info().rss / (1024**2)  # MB
        
        # Time the operation
        start_time = time.time()
        
        try:
            result = component_func(df_sample, *args)
            success = True
        except Exception as e:
            print(f"❌ Error in {component_name}: {e}")
            result = None
            success = False
        
        end_time = time.time()
        
        # Memory after
        memory_after = process.memory_info().rss / (1024**2)  # MB
        
        # Calculate metrics
        processing_time = end_time - start_time
        memory_used = memory_after - memory_before
        throughput = len(df_sample) / processing_time if processing_time > 0 else 0
        
        # Store metrics
        self.metrics['processing_times'][component_name] = processing_time
        self.metrics['memory_usage'][component_name] = memory_used
        self.metrics['throughput'][component_name] = throughput
        
        print(f"   ⏱️  Time: {processing_time:.2f}s")
        print(f"   💾 Memory: {memory_used:.1f}MB")
        print(f"   🚀 Throughput: {throughput:.1f} reviews/sec")
        
        return result, success
    
    def analyze_pipeline_performance(self, df_test, sample_sizes=[10, 50, 100]):
        """Comprehensive pipeline performance analysis"""
        
        print(f"📊 COMPREHENSIVE PIPELINE PERFORMANCE ANALYSIS")
        print("=" * 60)
        
        results = {}
        
        # Test different sample sizes
        for size in sample_sizes:
            if size > len(df_test):
                continue
            
            print(f"\n🔍 Testing with {size} reviews...")
            sample_df = df_test.head(size).copy()
            
            size_results = {}
            
            # Test each component
            components = [
                ('Basic Preprocessing', self._test_basic_preprocessing),
                ('Advanced Features', self._test_advanced_features),
                ('Policy Detection', self._test_policy_detection),
                ('Topic Modeling', self._test_topic_modeling),
                ('Keyword Extraction', self._test_keyword_extraction),
                ('Similarity Analysis', self._test_similarity_analysis)
            ]
            
            for comp_name, comp_func in components:
                result, success = self.benchmark_component_performance(
                    sample_df, f"{comp_name}_{size}", comp_func, sample_df
                )
                size_results[comp_name] = {
                    'success': success,
                    'time': self.metrics['processing_times'].get(f"{comp_name}_{size}", 0),
                    'memory': self.metrics['memory_usage'].get(f"{comp_name}_{size}", 0),
                    'throughput': self.metrics['throughput'].get(f"{comp_name}_{size}", 0)
                }
            
            results[size] = size_results
        
        # Analyze scalability
        self._analyze_scalability(results, sample_sizes)
        
        return results
    
    def _test_basic_preprocessing(self, df):
        """Test basic preprocessing performance"""
        from data_processing import TextPreprocessor, PolicyFeatureExtractor
        
        processor = TextPreprocessor()
        policy_extractor = PolicyFeatureExtractor()
        
        # Basic preprocessing
        df_processed = processor.preprocess_dataframe(df, 'review_text')
        
        # Basic policy features
        df_policy = policy_extractor.extract_features_dataframe(df_processed, 'cleaned_text')
        
        return df_policy
    
    def _test_advanced_features(self, df):
        """Test advanced feature extraction performance"""
        extractor = AdvancedNLPFeatureExtractor()
        result = extractor.process_dataframe(df, 'review_text')
        return result
    
    def _test_policy_detection(self, df):
        """Test policy detection performance"""
        detector = PolicyViolationDetector()
        result = detector.detect_violations_rules(df)
        return result
    
    def _test_topic_modeling(self, df):
        """Test topic modeling performance"""
        modeler = RestaurantTopicModeler(n_topics=5)
        result, summary = modeler.analyze_dataset_topics(df)
        return result
    
    def _test_keyword_extraction(self, df):
        """Test keyword extraction performance"""
        extractor = RestaurantKeywordExtractor()
        result = extractor.analyze_review_keywords(df)
        return result
    
    def _test_similarity_analysis(self, df):
        """Test similarity analysis performance"""
        analyzer = ReviewSimilarityAnalyzer()
        result = analyzer.generate_similarity_report(df)
        return result
    
    def _analyze_scalability(self, results, sample_sizes):
        """Analyze scalability patterns"""
        
        print(f"\n📈 SCALABILITY ANALYSIS")
        print("=" * 40)
        
        for component in ['Basic Preprocessing', 'Advanced Features', 'Policy Detection']:
            times = [results[size][component]['time'] for size in sample_sizes if size in results]
            sizes = [size for size in sample_sizes if size in results]
            
            if len(times) >= 2:
                # Calculate scaling factor
                time_ratio = times[-1] / times[0] if times[0] > 0 else float('inf')
                size_ratio = sizes[-1] / sizes[0]
                scaling_efficiency = size_ratio / time_ratio if time_ratio > 0 else 0
                
                print(f"   {component}:")
                print(f"     Time scaling: {time_ratio:.2f}x for {size_ratio:.0f}x data")
                print(f"     Efficiency: {scaling_efficiency:.2f}")
                
                # Identify bottlenecks
                if scaling_efficiency < 0.5:
                    self.metrics['bottlenecks'].append(f"{component} scales poorly")
    
    def optimize_memory_usage(self, df):
        """Optimize memory usage for large datasets"""
        
        print(f"💾 MEMORY OPTIMIZATION ANALYSIS")
        print("=" * 40)
        
        initial_memory = df.memory_usage(deep=True).sum() / (1024**2)  # MB
        print(f"   Initial DataFrame memory: {initial_memory:.1f}MB")
        
        optimizations = []
        optimized_df = df.copy()
        
        # Optimize string columns
        for col in optimized_df.columns:
            if optimized_df[col].dtype == 'object':
                try:
                    # Try to convert to category if many repeats
                    unique_ratio = optimized_df[col].nunique() / len(optimized_df)
                    if unique_ratio < 0.5:  # Less than 50% unique values
                        optimized_df[col] = optimized_df[col].astype('category')
                        optimizations.append(f"Converted {col} to category")
                except:
                    pass
        
        # Optimize numeric columns
        for col in optimized_df.columns:
            if optimized_df[col].dtype in ['int64', 'float64']:
                col_min = optimized_df[col].min()
                col_max = optimized_df[col].max()
                
                # Downcast integers
                if optimized_df[col].dtype == 'int64':
                    if col_min >= -128 and col_max <= 127:
                        optimized_df[col] = optimized_df[col].astype('int8')
                        optimizations.append(f"Downcasted {col} to int8")
                    elif col_min >= -32768 and col_max <= 32767:
                        optimized_df[col] = optimized_df[col].astype('int16')
                        optimizations.append(f"Downcasted {col} to int16")
                    elif col_min >= -2147483648 and col_max <= 2147483647:
                        optimized_df[col] = optimized_df[col].astype('int32')
                        optimizations.append(f"Downcasted {col} to int32")
                
                # Downcast floats
                elif optimized_df[col].dtype == 'float64':
                    optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='float')
                    optimizations.append(f"Downcasted {col} to float32")
        
        final_memory = optimized_df.memory_usage(deep=True).sum() / (1024**2)  # MB
        memory_saved = initial_memory - final_memory
        reduction_percent = (memory_saved / initial_memory) * 100
        
        print(f"   Optimized DataFrame memory: {final_memory:.1f}MB")
        print(f"   Memory saved: {memory_saved:.1f}MB ({reduction_percent:.1f}%)")
        print(f"   Optimizations applied: {len(optimizations)}")
        
        for opt in optimizations:
            print(f"     • {opt}")
        
        return optimized_df, memory_saved
    
    def suggest_parallel_processing(self, estimated_total_reviews):
        """Suggest parallel processing strategies"""
        
        print(f"\n🔄 PARALLEL PROCESSING RECOMMENDATIONS")
        print("=" * 50)
        
        cpu_count = self.system_info['cpu_count']
        available_memory = self.system_info['memory_available']
        
        # Estimate memory per review (from benchmarks)
        avg_memory_per_review = 2.0  # MB (conservative estimate)
        
        # Calculate optimal batch size
        optimal_batch_size = min(
            int(available_memory * 1024 / avg_memory_per_review * 0.7),  # 70% of available memory
            estimated_total_reviews // cpu_count if estimated_total_reviews > cpu_count else estimated_total_reviews
        )
        
        num_batches = (estimated_total_reviews + optimal_batch_size - 1) // optimal_batch_size
        
        print(f"   Estimated total reviews: {estimated_total_reviews:,}")
        print(f"   Available CPU cores: {cpu_count}")
        print(f"   Available memory: {available_memory:.1f}GB")
        print(f"   Recommended batch size: {optimal_batch_size:,} reviews")
        print(f"   Number of batches: {num_batches}")
        
        # Processing strategy recommendation
        if estimated_total_reviews < 1000:
            strategy = "Sequential processing (small dataset)"
        elif estimated_total_reviews < 10000:
            strategy = "Thread-based parallelism"
        else:
            strategy = "Process-based parallelism with batching"
        
        print(f"   Recommended strategy: {strategy}")
        
        # Estimated processing time
        if self.metrics['throughput']:
            avg_throughput = np.mean(list(self.metrics['throughput'].values()))
            estimated_time = estimated_total_reviews / (avg_throughput * cpu_count * 0.8)  # 80% efficiency
            
            if estimated_time < 60:
                time_str = f"{estimated_time:.1f} seconds"
            elif estimated_time < 3600:
                time_str = f"{estimated_time/60:.1f} minutes"
            else:
                time_str = f"{estimated_time/3600:.1f} hours"
            
            print(f"   Estimated processing time: {time_str}")
        
        return {
            'strategy': strategy,
            'batch_size': optimal_batch_size,
            'num_batches': num_batches,
            'cpu_utilization': cpu_count
        }
    
    def generate_optimization_report(self, df_test, estimated_total_reviews=10000):
        """Generate comprehensive optimization report"""
        
        print(f"📋 GENERATING COMPREHENSIVE OPTIMIZATION REPORT")
        print("=" * 60)
        
        # Run performance analysis
        performance_results = self.analyze_pipeline_performance(df_test)
        
        # Memory optimization
        optimized_df, memory_saved = self.optimize_memory_usage(df_test)
        
        # Parallel processing recommendations
        parallel_rec = self.suggest_parallel_processing(estimated_total_reviews)
        
        # Generate final recommendations
        recommendations = self._generate_optimization_recommendations()
        
        report = {
            'system_info': self.system_info,
            'performance_analysis': performance_results,
            'memory_optimization': {
                'memory_saved_mb': memory_saved,
                'optimization_count': len(recommendations)
            },
            'parallel_processing': parallel_rec,
            'bottlenecks': self.metrics['bottlenecks'],
            'recommendations': recommendations,
            'summary': {
                'overall_performance': self._calculate_overall_performance(),
                'scalability_rating': self._calculate_scalability_rating(),
                'optimization_potential': self._calculate_optimization_potential()
            }
        }
        
        return report
    
    def _generate_optimization_recommendations(self):
        """Generate specific optimization recommendations"""
        
        recommendations = []
        
        # Performance-based recommendations
        if self.metrics['bottlenecks']:
            recommendations.extend([
                "Implement batch processing for memory-intensive operations",
                "Consider feature selection to reduce computational overhead",
                "Use parallel processing for independent operations"
            ])
        
        # Memory-based recommendations
        if self.system_info['memory_available'] < 4.0:  # Less than 4GB available
            recommendations.append("Consider processing in smaller batches due to memory constraints")
        
        # CPU-based recommendations
        if self.system_info['cpu_count'] > 4:
            recommendations.append("Utilize multiprocessing for CPU-intensive operations")
        
        # General recommendations
        recommendations.extend([
            "Cache preprocessed features to avoid recomputation",
            "Use vectorized operations where possible",
            "Implement early stopping for iterative algorithms",
            "Consider using approximate algorithms for large-scale processing"
        ])
        
        return recommendations
    
    def _calculate_overall_performance(self):
        """Calculate overall performance score"""
        if not self.metrics['throughput']:
            return 0.0
        
        avg_throughput = np.mean(list(self.metrics['throughput'].values()))
        
        # Normalize to 0-1 scale (100 reviews/sec = 1.0)
        return min(avg_throughput / 100, 1.0)
    
    def _calculate_scalability_rating(self):
        """Calculate scalability rating"""
        bottleneck_penalty = len(self.metrics['bottlenecks']) * 0.2
        return max(1.0 - bottleneck_penalty, 0.0)
    
    def _calculate_optimization_potential(self):
        """Calculate optimization potential"""
        # Based on number of recommendations and bottlenecks
        potential = (len(self.metrics['bottlenecks']) + 
                    len(self.metrics['optimization_suggestions'])) / 10
        return min(potential, 1.0)


# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing NLP Performance Optimizer...")
    
    # Create test data
    test_reviews = [
        "Amazing food and great service!",
        "The pasta was delicious and perfectly cooked.",
        "Terrible experience with cold food.",
        "Beautiful ambiance and romantic atmosphere.",
        "Overpriced for the portion size.",
        "Quick service and friendly staff.",
        "Best pizza I've ever had!",
        "Disappointing meal and rude waiter.",
        "Perfect location with easy parking.",
        "Fresh ingredients and amazing flavors."
    ] * 5  # Multiply to create larger test set
    
    test_df = pd.DataFrame({
        'review_text': test_reviews,
        'rating': [5, 4, 1, 5, 2, 4, 5, 1, 4, 5] * 5,
        'user_id': [f'user{i}' for i in range(50)]
    })
    
    # Initialize optimizer
    optimizer = NLPPerformanceOptimizer()
    
    # Generate optimization report
    report = optimizer.generate_optimization_report(test_df, estimated_total_reviews=5000)
    
    print(f"\n📊 OPTIMIZATION REPORT SUMMARY:")
    print(f"   Overall Performance: {report['summary']['overall_performance']:.2f}")
    print(f"   Scalability Rating: {report['summary']['scalability_rating']:.2f}")
    print(f"   Optimization Potential: {report['summary']['optimization_potential']:.2f}")
    
    print(f"\n🚀 PARALLEL PROCESSING RECOMMENDATIONS:")
    print(f"   Strategy: {report['parallel_processing']['strategy']}")
    print(f"   Batch Size: {report['parallel_processing']['batch_size']:,}")
    
    print(f"\n💡 TOP RECOMMENDATIONS:")
    for i, rec in enumerate(report['recommendations'][:3], 1):
        print(f"   {i}. {rec}")
    
    print("\n✅ Performance Optimizer test complete!")
