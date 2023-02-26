import os

fileNames = sorted(os.listdir("./model"))
errorFiles = []

vaePath = None
while vaePath not in ('1', '2', '3'):
    print("Type the vae_path option\n")
    vaePath = input("[1] = Anything-v3.0 (Anime)\n[2] = StabilityAI (Realistic)\n[3] = None (Default)\n")

if vaePath == '1':
    vaePath = '--vae_path "Linaqruf/anything-v3.0/vae"'
elif vaePath == '2':
    vaePath = '--vae_path "stabilityai/sd-vae-ft-mse"'
else:
    vaePath = ""

for i in range(len(fileNames)):
    if not fileNames[i].endswith('.safetensors'):
        continue
    
    singleFile = fileNames[i].split("_")

    if not os.path.exists(f"./model/{singleFile[0]}"):
        print(f'Now converting {fileNames[i]} inside the folder {singleFile[0]}')
        os.system(f'python conv_sd_to_onnx.py --model_path "./model/{fileNames[i]}" --output_path "./model/{singleFile[0]}" --ckpt-original-config-file v1-inference.yaml {vaePath} --fp16 --attention-slicing max')
        continue
