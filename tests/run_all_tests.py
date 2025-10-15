#!/usr/bin/env python3
"""
Main test runner for the CUDA-Triton-TileLang comparison project.
This script runs all tests and generates comprehensive reports.
"""

import os
import sys
import argparse
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from accuracy_test import AccuracyTester
from performance_test import PerformanceTester
from visualization import ResultsVisualizer

def main():
    parser = argparse.ArgumentParser(description='Run all tests for CUDA-Triton-TileLang comparison')
    parser.add_argument('--accuracy-only', action='store_true', help='Run only accuracy tests')
    parser.add_argument('--performance-only', action='store_true', help='Run only performance tests')
    parser.add_argument('--device', default='cuda', help='Device to run tests on (cuda/cpu)')
    parser.add_argument('--results-dir', default='results', help='Directory to save results')
    
    args = parser.parse_args()
    
    print("CUDA-Triton-TileLang Comparison Test Suite")
    print("=" * 50)
    print(f"Device: {args.device}")
    print(f"Results directory: {args.results_dir}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Initialize testers
    accuracy_tester = AccuracyTester(device=args.device)
    performance_tester = PerformanceTester(device=args.device)
    visualizer = ResultsVisualizer(results_dir=args.results_dir)
    
    results = {}
    
    # Run accuracy tests
    if not args.performance_only:
        print("\nRunning accuracy tests...")
        try:
            accuracy_tester.run_all_tests()
            print("✓ Accuracy tests completed successfully")
        except Exception as e:
            print(f"✗ Accuracy tests failed: {e}")
    
    # Run performance tests
    if not args.accuracy_only:
        print("\nRunning performance tests...")
        try:
            performance_results = performance_tester.run_all_tests()
            results.update(performance_results)
            print("✓ Performance tests completed successfully")
        except Exception as e:
            print(f"✗ Performance tests failed: {e}")
    
    # Generate visualizations and reports
    if results:
        print("\nGenerating visualizations and reports...")
        try:
            # Plot performance comparisons
            for operation in ['matrix_add', 'matrix_mul', 'softmax']:
                if operation in results:
                    visualizer.plot_performance_comparison(results[operation], operation)
            
            # Generate summary report
            visualizer.generate_summary_report(results, {})
            print("✓ Visualizations and reports generated successfully")
        except Exception as e:
            print(f"✗ Visualization generation failed: {e}")
    
    print("\n" + "=" * 50)
    print("Test suite completed!")
    print(f"Results saved to: {args.results_dir}")

if __name__ == "__main__":
    main()
