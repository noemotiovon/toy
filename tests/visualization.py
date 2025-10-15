import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

class ResultsVisualizer:
    """Visualize test results with charts and graphs"""
    
    def __init__(self, results_dir="results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
    def plot_performance_comparison(self, results, operation_name):
        """Plot performance comparison for a specific operation"""
        plt.figure(figsize=(12, 8))
        
        # Extract data for plotting
        frameworks = []
        sizes = []
        times = []
        errors = []
        
        for key, value in results.items():
            if operation_name in key:
                parts = key.split('_')
                framework = parts[0]
                size = '_'.join(parts[1:])
                
                frameworks.append(framework)
                sizes.append(size)
                times.append(value['mean'] * 1000)  # Convert to ms
                errors.append(value['std'] * 1000)
        
        # Create DataFrame
        df = pd.DataFrame({
            'Framework': frameworks,
            'Size': sizes,
            'Time (ms)': times,
            'Error (ms)': errors
        })
        
        # Create bar plot
        plt.subplot(2, 1, 1)
        sns.barplot(data=df, x='Size', y='Time (ms)', hue='Framework')
        plt.title(f'{operation_name.title()} Performance Comparison')
        plt.ylabel('Time (ms)')
        plt.xticks(rotation=45)
        
        # Create speedup plot
        plt.subplot(2, 1, 2)
        pytorch_times = df[df['Framework'] == 'PyTorch'].set_index('Size')['Time (ms)']
        speedup_data = []
        
        for framework in df['Framework'].unique():
            if framework != 'PyTorch':
                framework_times = df[df['Framework'] == framework].set_index('Size')['Time (ms)']
                speedup = pytorch_times / framework_times
                speedup_data.append({
                    'Framework': framework,
                    'Size': speedup.index,
                    'Speedup': speedup.values
                })
        
        if speedup_data:
            speedup_df = pd.concat([pd.DataFrame(data) for data in speedup_data])
            sns.barplot(data=speedup_df, x='Size', y='Speedup', hue='Framework')
            plt.title(f'{operation_name.title()} Speedup vs PyTorch')
            plt.ylabel('Speedup (x)')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f'{operation_name}_performance.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_accuracy_comparison(self, accuracy_results):
        """Plot accuracy comparison"""
        plt.figure(figsize=(10, 6))
        
        operations = list(accuracy_results.keys())
        frameworks = ['Triton', 'TileLang']
        
        x = np.arange(len(operations))
        width = 0.35
        
        for i, framework in enumerate(frameworks):
            errors = [accuracy_results[op].get(framework, float('inf')) for op in operations]
            plt.bar(x + i*width, errors, width, label=framework, alpha=0.8)
        
        plt.xlabel('Operations')
        plt.ylabel('Max Absolute Error')
        plt.title('Accuracy Comparison (Lower is Better)')
        plt.xticks(x + width/2, operations)
        plt.legend()
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'accuracy_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
    def generate_summary_report(self, performance_results, accuracy_results):
        """Generate a summary report"""
        report_path = os.path.join(self.results_dir, 'summary_report.md')
        
        with open(report_path, 'w') as f:
            f.write("# Performance and Accuracy Summary Report\n\n")
            
            f.write("## Performance Results\n\n")
            for operation, results in performance_results.items():
                f.write(f"### {operation.title()}\n\n")
                f.write("| Framework | Size | Time (ms) | Error (ms) |\n")
                f.write("|-----------|------|-----------|------------|\n")
                
                for key, value in results.items():
                    parts = key.split('_')
                    framework = parts[0]
                    size = '_'.join(parts[1:])
                    f.write(f"| {framework} | {size} | {value['mean']*1000:.2f} | {value['std']*1000:.2f} |\n")
                f.write("\n")
            
            f.write("## Accuracy Results\n\n")
            f.write("| Operation | Triton Error | TileLang Error |\n")
            f.write("|-----------|--------------|----------------|\n")
            
            for operation, results in accuracy_results.items():
                triton_error = results.get('Triton', 'N/A')
                tilelang_error = results.get('TileLang', 'N/A')
                f.write(f"| {operation} | {triton_error} | {tilelang_error} |\n")
            
        print(f"Summary report saved to {report_path}")

if __name__ == "__main__":
    # Example usage
    visualizer = ResultsVisualizer()
    
    # Example results (replace with actual results)
    example_performance = {
        'matrix_add': {
            'PyTorch_1024x1024': {'mean': 0.001, 'std': 0.0001},
            'Triton_1024x1024': {'mean': 0.0008, 'std': 0.0001},
            'TileLang_1024x1024': {'mean': 0.0009, 'std': 0.0001},
        }
    }
    
    example_accuracy = {
        'matrix_add': {'Triton': 1e-6, 'TileLang': 1e-6},
        'matrix_mul': {'Triton': 1e-5, 'TileLang': 1e-5},
        'softmax': {'Triton': 1e-6, 'TileLang': 1e-6},
    }
    
    visualizer.plot_performance_comparison(example_performance['matrix_add'], 'matrix_add')
    visualizer.plot_accuracy_comparison(example_accuracy)
    visualizer.generate_summary_report(example_performance, example_accuracy)
