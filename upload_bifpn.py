import os
import sys

try:
    from lightning_sdk import Studio
except ImportError:
    print("Error: lightning_sdk is not installed. Please run 'pip install lightning-sdk'")
    sys.exit(1)

print("="*50)

studio_name = "model-training-devbox"
file_to_upload = "lightning_upload_bifpn.zip"
destination = f"/home/zeus/{file_to_upload}"

if not os.path.exists(file_to_upload):
    print(f"Error: Could not find {file_to_upload} in the current directory.")
    sys.exit(1)

print(f"\nConnecting to Studio '{studio_name}'...")
try:
    studio = Studio(
        name="model-development-project-teamspace/model-training-devbox"
    )
    
    print(f"Uploading {file_to_upload} to {destination}...")
    studio.upload_file(file_to_upload, destination)
    
    print("\n[SUCCESS] Upload successful!")
    print("Executing commands on the studio...")
    
    # Run the setup and training commands remotely
    studio.run(f"unzip -o {destination} -d /teamspace/studios/this_studio")
    studio.run("chmod +x /teamspace/studios/this_studio/wrapper_bifpn.sh")
    
    # Start the training as a background job using nohup so we don't block
    studio.run("nohup /teamspace/studios/this_studio/wrapper_bifpn.sh > /teamspace/studios/this_studio/bifpn_train.log 2>&1 &")
    print("\n[SUCCESS] Training job started in the background on Lightning AI Studio!")
    print("You can monitor the training by checking bifpn_train.log on the studio.")

except Exception as e:
    print("\n[FAILED] Upload or execution failed. Error Details:")
    print(e)
