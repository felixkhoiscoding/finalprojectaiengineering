# Project Cleanup Script
# Run this to organize the Final Project directory

import os
import shutil
from pathlib import Path

print("="*60)
print("FINAL PROJECT CLEANUP")
print("="*60)

project_root = Path(__file__).parent

# Create archive folder for old files
archive = project_root / "_archive_old_scripts"
archive.mkdir(exist_ok=True)

# Files to archive (move out of root)
files_to_archive = [
    "WPU101704.xlsx",  # Duplicate - already in data/raw
    "generate_forecasts_clean.py",  # Backup version
    "fix_lstm_data.py",  # Temporary script
    "test_arima_fix.py",  # Test script
    "test_lstm.py",  # Test script
    "test_step1.py",  # Test script
    "test_step2.py",  # Test script
    "test_step3.py",  # Test script
    "test_step4.py",  # Test script
    "test_step5.py",  # Test script
    "verify_project.py",  # Temporary script
]

print("\nMoving old scripts to _archive_old_scripts/...\n")

for filename in files_to_archive:
    source = project_root / filename
    if source.exists():
        dest = archive / filename
        shutil.move(str(source), str(dest))
        print(f"  Archived: {filename}")
    else:
        print(f"  Skip (not found): {filename}")

print("\n" + "="*60)
print("CLEANUP COMPLETE!")
print("="*60)
print("\nRoot directory now contains only:")
print("  - streamlit_app.py (main app)")
print("  - generate_forecasts.py (forecast generation)")
print("  - README.md (documentation)")
print("  - requirements.txt (dependencies)")
print("  - config/ src/ data/ models/ notebooks/ results/ (folders)")
print("\nOld files moved to: _archive_old_scripts/")
print("="*60)
