scp ultralytics_block.py s_01kyp2s7rn5thzg7jnn5y3m6v1@ssh.lightning.ai:/teamspace/studios/this_studio/
scp ultralytics_tasks.py s_01kyp2s7rn5thzg7jnn5y3m6v1@ssh.lightning.ai:/teamspace/studios/this_studio/
scp patch_stacked_v2_remote.py s_01kyp2s7rn5thzg7jnn5y3m6v1@ssh.lightning.ai:/teamspace/studios/this_studio/
scp yolov8n-bifpn-cbam.yaml s_01kyp2s7rn5thzg7jnn5y3m6v1@ssh.lightning.ai:/teamspace/studios/this_studio/
scp train_stacked_v2.py s_01kyp2s7rn5thzg7jnn5y3m6v1@ssh.lightning.ai:/teamspace/studios/this_studio/
scp wrapper_stacked_v2.sh s_01kyp2s7rn5thzg7jnn5y3m6v1@ssh.lightning.ai:/teamspace/studios/this_studio/

ssh s_01kyp2s7rn5thzg7jnn5y3m6v1@ssh.lightning.ai "cd /teamspace/studios/this_studio/ && /home/zeus/miniconda3/envs/cloudspace/bin/python patch_stacked_v2_remote.py"

ssh s_01kyp2s7rn5thzg7jnn5y3m6v1@ssh.lightning.ai "cd /teamspace/studios/this_studio/ && dos2unix wrapper_stacked_v2.sh && chmod +x wrapper_stacked_v2.sh && ./wrapper_stacked_v2.sh"

echo "Deployed and started training Stacked V2!"
