import os

fileNames = sorted(os.listdir("./model"))

vaePath = None
while vaePath not in ('1', '2', '3'):
    print("Type the vae_path option\n")
    vaePath = input(
        "[1] = Anything-v3.0 (Anime)\n"
        "[2] = StabilityAI (Realistic)\n"
        "[3] = None (Default)\n"
    )
clipSkip = None
while clipSkip not in ('1', '2', '3', '4'):
    print("Type the ClipSkip option\n")
    clipSkip = input(
        "[1] = ClipSkip 2\n"
        "[2] = ClipSkip 3\n"
        "[3] = ClipSkip 4\n"
        "[4] = None (Default)\n"
    )

if vaePath == '1':
    vaePath = '--vae_path "Linaqruf/anything-v3.0/vae" '
elif vaePath == '2':
    vaePath = '--vae_path "stabilityai/sd-vae-ft-mse" '
else:
    vaePath = ""

if clipSkip == '1':
    clipSkip = '--clip-skip 2 '
elif clipSkip == '2':
    clipSkip = '--clip-skip 3 '
elif clipSkip == '3':
    clipSkip = '--clip-skip 4 '
else:
    clipSkip = ""

for i in range(len(fileNames)):
    if not fileNames[i].endswith('.safetensors'):
        continue

    singleFile = fileNames[i].split("_")

    if not os.path.exists(f"./model/{singleFile[0]}"):
        print(
            f'Now converting {fileNames[i]} inside the folder {singleFile[0]}'
        )
        os.system(
            f'python conv_sd_to_onnx.py '
            f'--model_path "./model/{fileNames[i]}" '
            f'--output_path "./model/{singleFile[0]}" '
            f'--ckpt-original-config-file v1-inference.yaml '
            f'{vaePath}'
            f'{clipSkip}'
            f'--fp16 '
            f'--attention-slicing max'
        )
        continue