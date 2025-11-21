#!/usr/bin/env python3
"""
Script to combine all training log files from 33bus, 57bus, and 118bus systems
into a single .txt file for upload to Google AI Studio.
"""

import os
from pathlib import Path

def combine_logs():
    # Base directory
    base_dir = Path(__file__).parent / "experimental_results" / "run_20251117_042856"
    
    # Output file
    output_file = base_dir / "all_training_logs_combined.txt"
    
    # Bus systems to process
    bus_systems = ["33bus", "57bus", "118bus"]
    
    # Collect all log files
    log_files = []
    for bus in bus_systems:
        log_dir = base_dir / bus / "log"
        if log_dir.exists():
            for log_file in sorted(log_dir.glob("*.log")):
                # Skip combined_logs.txt if it exists
                if log_file.name != "combined_logs.txt":
                    log_files.append((bus, log_file))
    
    # Write combined logs
    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.write("=" * 100 + "\n")
        outfile.write("COMBINED TRAINING LOGS - ALL BUS SYSTEMS\n")
        outfile.write(f"Generated from: {base_dir}\n")
        outfile.write("=" * 100 + "\n\n")
        
        for bus, log_file in log_files:
            # Write separator
            outfile.write("\n" + "=" * 100 + "\n")
            outfile.write(f"BUS SYSTEM: {bus.upper()} | MODEL: {log_file.stem}\n")
            outfile.write(f"FILE: {log_file.name}\n")
            outfile.write("=" * 100 + "\n\n")
            
            # Read and write log content
            try:
                with open(log_file, 'r', encoding='utf-8') as infile:
                    content = infile.read()
                    outfile.write(content)
                    # Add newline if file doesn't end with one
                    if content and not content.endswith('\n'):
                        outfile.write('\n')
            except Exception as e:
                outfile.write(f"ERROR reading {log_file}: {e}\n")
            
            outfile.write("\n" + "-" * 100 + "\n\n")
        
        outfile.write("=" * 100 + "\n")
        outfile.write("END OF COMBINED LOGS\n")
        outfile.write("=" * 100 + "\n")
    
    print(f"✓ Successfully combined {len(log_files)} log files")
    print(f"✓ Output saved to: {output_file}")
    print(f"\nFiles combined:")
    for bus, log_file in log_files:
        print(f"  - {bus}/{log_file.name}")
    
    return output_file

if __name__ == "__main__":
    combine_logs()

