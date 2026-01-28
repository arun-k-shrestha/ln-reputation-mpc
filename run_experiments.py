import subprocess
import configparser

# Define malicious percentages to test
malicious_percentages = [0.0, 0.05,0.1]

for percent in malicious_percentages:
    print(f"\n{'='*50}")
    print(f"Running with {percent*100}% malicious nodes")
    print(f"{'='*50}\n")
    
    # Update config file
    config = configparser.ConfigParser()
    config.read('config.ini')
    config['General']['malicious_percent'] = str(percent)
    
    with open('config.ini', 'w') as f:
        config.write(f)
    
    # Run the main script
    subprocess.run(['python3', 'run_simulation.py'])
    
    print(f"\nCompleted {percent*100}% malicious nodes experiment")

print("\n" + "="*50)
print("All experiments completed!")