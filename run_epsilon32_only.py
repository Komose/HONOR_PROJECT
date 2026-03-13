"""
Quick Test: epsilon=32/255 only (Extreme Comparison)
=====================================================
Run FGSM and PGD with extreme epsilon value for comparison
"""

import subprocess
import sys
from datetime import datetime

# Modify epsilon list temporarily for this run
print("="*70)
print("EXTREME EPSILON TEST: ε=32/255")
print("="*70)
print("\nThis script will test ONLY epsilon=32/255")
print("for FGSM and PGD algorithms.\n")

# Create temporary modified script
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir = f'results/epsilon32_extreme_{timestamp}'

# Modify the main script to only test 32/255
import fileinput
import shutil

# Backup original file
shutil.copy('run_multi_algorithm_experiments.py',
            'run_multi_algorithm_experiments.py.backup')

# Modify epsilon values
with fileinput.FileInput('run_multi_algorithm_experiments.py', inplace=True) as file:
    for line in file:
        if 'EPSILON_VALUES = [1/255, 2/255, 4/255, 16/255, 32/255]' in line:
            print('    EPSILON_VALUES = [32/255]  # Temporary: extreme test only')
        else:
            print(line, end='')

# Run experiment
print("Starting experiment...")
cmd = [
    sys.executable,
    'run_multi_algorithm_experiments.py',
    '--algorithms', 'fgsm,pgd',
    '--samples', '200',
    '--batch_size', '1',
    '--enable_random_patch',
    '--output_dir', output_dir
]

try:
    subprocess.run(cmd, check=True)
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETED!")
    print(f"Results: {output_dir}/")
    print("="*70)
finally:
    # Restore original file
    shutil.move('run_multi_algorithm_experiments.py.backup',
                'run_multi_algorithm_experiments.py')
    print("\nOriginal script restored.")
