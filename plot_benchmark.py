import matplotlib.pyplot as plt
import numpy as np

def parse_bench_data(filename):
    """Parse benchmark data from bench.txt file"""
    
    # Data for N=128
    n128_data = {
        'iter': [100, 1000, 10000],
        '1660Super': [1.283646, 12.565453, 126.336135],
        'rtx4080Super': [0.155406, 1.557794, 15.676164],
        '128CPU': [2.404, 24.247, 244.960]
    }
    
    # Data for N=256
    n256_data = {
        'iter': [100, 1000, 10000],
        '1660Super': [13.569758, 135.586401, 1356.530211],
        'rtx4080Super': [1.598243, 16.176335, 159.067719],
        '128CPU': [22.730, 252.177, 2523.142]  # Missing data for 10000 iterations
    }
    
    return n128_data, n256_data

def create_plots():
    """Create benchmark plots for N=128 and N=256"""
    
    # Parse the data
    n128_data, n256_data = parse_bench_data('bench.txt')
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot for N=128
    ax1.plot(n128_data['iter'], n128_data['1660Super'], 'o-', label='1660 Super', linewidth=2, markersize=8)
    ax1.plot(n128_data['iter'], n128_data['rtx4080Super'], 's-', label='RTX 4080 Super', linewidth=2, markersize=8)
    ax1.plot(n128_data['iter'], n128_data['128CPU'], '^-', label='128 CPUs', linewidth=2, markersize=8)
    
    ax1.set_xlabel('Iterations', fontsize=12)
    ax1.set_ylabel('Time (seconds)', fontsize=12)
    ax1.set_title('Benchmark Results - N=128', fontsize=14, fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Plot for N=256
    
    ax2.plot(n256_data['iter'], n256_data['1660Super'], 'o-', label='1660 Super', linewidth=2, markersize=8)
    ax2.plot(n256_data['iter'], n256_data['rtx4080Super'], 's-', label='RTX 4080 Super', linewidth=2, markersize=8)
    ax2.plot(n256_data['iter'], n256_data['128CPU'], '^-', label='128 CPUs', linewidth=2, markersize=8)
    
    ax2.set_xlabel('Iterations', fontsize=12)
    ax2.set_ylabel('Time (seconds)', fontsize=12)
    ax2.set_title('Benchmark Results - N=256', fontsize=14, fontweight='bold')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('benchmark_comparison.png', dpi=300, bbox_inches='tight')
    
    # Show the plot
    plt.show()
    
    print("Benchmark plots created successfully!")
    print("Plot saved as 'benchmark_comparison.png'")

if __name__ == "__main__":
    create_plots()
