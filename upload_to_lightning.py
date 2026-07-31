import os
import sys
import getpass

try:
    from lightning_sdk import Studio
except ImportError:
    print("Error: lightning_sdk is not installed. Please run 'pip install lightning-sdk'")
    sys.exit(1)

print("="*50)
# Credentials are read automatically from ~/.lightning/credentials.json

studio_name = "model-training-devbox"

file_to_upload = "lightning_upload_with_data.zip"
destination = f"/home/zeus/{file_to_upload}"

if not os.path.exists(file_to_upload):
    print(f"Error: Could not find {file_to_upload} in the current directory.")
    sys.exit(1)

print(f"\nConnecting to Studio '{studio_name}'...")
try:
    studio = Studio(
        name="model-development-project-teamspace/model-training-devbox"
    )
    
    # Upload the file
    print(f"Uploading {file_to_upload} to {destination}...")
    
    studio.upload_file(file_to_upload, destination)
    
    print("\n[SUCCESS] Upload successful!")
    print("You can now go to your Lightning Studio and run:")
    print(f"unzip {file_to_upload}")
    print("python run_rfa_experiment.py")

except Exception as e:
    print("\n[FAILED] Upload failed. Please check:")
    print("1. Is your API key correct?")
    print("2. Is the Studio name spelled exactly right?")
    print("3. Is the Studio currently 'Running' in your browser?")
    print(f"\nError Details: {e}")
