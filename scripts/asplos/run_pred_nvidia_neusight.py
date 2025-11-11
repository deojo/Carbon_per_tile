import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import sys
import torch
import subprocess

def cuda_rr():
  idx = cuda_rr.cuda_idx % cuda_rr.device_num
  cuda_rr.cuda_idx += 1
  return "cuda:"+str(idx)
cuda_rr.cuda_idx = 0
cuda_rr.device_num = torch.cuda.device_count()

jobs = []

def launch_task(model_name, mode, batch_size, seqlen, device, option="", blocking=True):
    p = subprocess.Popen(
        [
            "python3", "../pred.py",
            "--predictor_name", "neusight",
            "--predictor_path", "./data/predictor/MLP_WAVE",
            "--device_config_path", f"./data/device_configs/{device}.json",
            "--model_config_path", f"./data/DLmodel_configs/{model_name}.json",
            "--sequence_length", str(seqlen),
            "--batch_size", str(batch_size),
            "--execution_type", mode,
            "--tile_dataset_dir", "./data/dataset/train",
            "--result_dir", "./results",
            "--options", option,
            "--running_device", cuda_rr(),
        ], 
        stdout=sys.stdout, 
        stderr=sys.stdout, 
    )

    if blocking:
        p.wait()

    jobs.append(p)

def pred():

    # main inference
    #for device in ["NVIDIA_H100_80GB_HBM3", "NVIDIA_A100-PCIE-40GB", "Tesla_T4", "Tesla_V100-PCIE-32GB", "Tesla_P100-PCIE-16GB", "Tesla_P4", "NVIDIA_L4", "NVIDIA_B200",]:
    
    #for device in ["NVIDIA_B200", "NVIDIA_H100_80GB_HBM3", "NVIDIA_A100_80GB_PCIe"]:

    for device in ["NVIDIA_H100_80GB_HBM3"]:    
        #launch_task(model_name="sheared_llama1_3B", mode="inf", batch_size=8, seqlen=512, device=device, blocking=False)
        #launch_task(model_name="sheared_llama2_7", mode="inf", batch_size=8, seqlen=512, device=device, blocking=False)
        #launch_task(model_name="llama2", mode="inf", batch_size=8, seqlen=512, device=device, blocking=False)
        #launch_task(model_name="sparse_llama", mode="inf", batch_size=8, seqlen=512, device=device, blocking=False)
        #launch_task(model_name="llama3", mode="inf", batch_size=8, seqlen=512, device=device, blocking=False)

        launch_task(model_name="bert_large", mode="inf", batch_size=8, seqlen=512, device=device, blocking=False)
        launch_task(model_name="bert_large", mode="inf", batch_size=16, seqlen=512, device=device, blocking=False)

        launch_task(model_name="gpt2_large", mode="inf", batch_size=4, seqlen=1024, device=device, blocking=False)
        launch_task(model_name="gpt2_large", mode="inf", batch_size=8, seqlen=1024, device=device, blocking=False)
        
        #launch_task(model_name="gpt3_xl", mode="inf", batch_size=2, seqlen=2048, device=device, blocking=False)
        #launch_task(model_name="gpt3_xl", mode="inf", batch_size=8, seqlen=2048, device=device, blocking=False)
        #launch_task(model_name="opt_13", mode="inf", batch_size=2, seqlen=2048, device=device, blocking=False)
        #launch_task(model_name="opt_13", mode="inf", batch_size=8, seqlen=2048, device=device, blocking=False)
        #launch_task(model_name="gpt3_27", mode="inf", batch_size=2, seqlen=2048, device=device, blocking=False)
        #launch_task(model_name="gpt3_27", mode="inf", batch_size=8, seqlen=2048, device=device, blocking=False)
        #launch_task(model_name="switchxl4", mode="inf", batch_size=1, seqlen=512, device=device, blocking=False)
        #launch_task(model_name="switchxl4", mode="inf", batch_size=2, seqlen=512, device=device, blocking=False)
        for p in jobs:
            p.wait()

    """
    # main train
    for device in ["NVIDIA_H100_80GB_HBM3", "NVIDIA_A100_80GB_PCIe", "Tesla_V100-PCIE-32GB", "NVIDIA_L4", "NVIDIA_B200",]:
        launch_task(model_name="bert_large", mode="train", batch_size=2, seqlen=512, device=device, blocking=False)
        launch_task(model_name="bert_large", mode="train", batch_size=8, seqlen=512, device=device, blocking=False)
        launch_task(model_name="gpt2_large", mode="train", batch_size=1, seqlen=1024, device=device, blocking=False)
        launch_task(model_name="gpt2_large", mode="train", batch_size=4, seqlen=1024, device=device, blocking=False)
        launch_task(model_name="gpt3_xl", mode="train", batch_size=1, seqlen=2048, device=device, blocking=False)
        launch_task(model_name="gpt3_xl", mode="train", batch_size=2, seqlen=2048, device=device, blocking=False)
        launch_task(model_name="opt_13", mode="train", batch_size=1, seqlen=2048, device=device, blocking=False)
        launch_task(model_name="opt_13", mode="train", batch_size=2, seqlen=2048, device=device, blocking=False)
        launch_task(model_name="gpt3_27", mode="train", batch_size=1, seqlen=2048, device=device, blocking=False)
        launch_task(model_name="gpt3_27", mode="train", batch_size=2, seqlen=2048, device=device, blocking=False)
        launch_task(model_name="switchxl4", mode="train", batch_size=1, seqlen=512, device=device, blocking=False)
        launch_task(model_name="switchxl4", mode="train", batch_size=2, seqlen=512, device=device, blocking=False)
        launch_task(model_name="llama2", mode="train", batch_size=1, seqlen=512, device=device, blocking=False)
        launch_task(model_name="llama3", mode="train", batch_size=1, seqlen=512, device=device, blocking=False)
        for p in jobs:
            p.wait()

    # fusion
    for device in ["NVIDIA_H100_80GB_HBM3", "NVIDIA_A100-PCIE-40GB", "NVIDIA_L4"]:
        launch_task(model_name="bert_large", mode="inf", batch_size=8, seqlen=512, device=device, blocking=False, option="fusion")
        launch_task(model_name="bert_large", mode="inf", batch_size=16, seqlen=512, device=device, blocking=False, option="fusion")
        launch_task(model_name="gpt2_large", mode="inf", batch_size=4, seqlen=1024, device=device, blocking=False, option="fusion")
        launch_task(model_name="gpt2_large", mode="inf", batch_size=8, seqlen=1024, device=device, blocking=False, option="fusion")
        for p in jobs:
            p.wait()
    

    # dist
    for device in ["NVIDIA_H100_80GB_HBM3", "NVIDIA_A100-SXM4-40GB", "Tesla_V100-SXM2-16GB"]:
        launch_task(model_name="bert", mode="train", batch_size=2, seqlen=512, device=device, blocking=False, option="pp4_1")
        launch_task(model_name="bert", mode="train", batch_size=1, seqlen=512, device=device, blocking=False, option="pp4_1")
        #launch_task(model_name="bert", mode="inf", batch_size=8, seqlen=512, device=device, blocking=False)
        #launch_task(model_name="bert", mode="inf", batch_size=16, seqlen=512, device=device, blocking=False)
        #launch_task(model_name="opt_13", mode="train", batch_size=1, seqlen=2048, device=device, blocking=False)
        #launch_task(model_name="opt_13", mode="train", batch_size=2, seqlen=2048, device=device, blocking=False)
        #launch_task(model_name="llama2", mode="train", batch_size=1, seqlen=2048, device=device, blocking=False, option="tp4")
        #launch_task(model_name="llama3", mode="train", batch_size=1, seqlen=2048, device=device, blocking=False, option="tp4")

        #launch_task(model_name="gpt2_large", mode="train", batch_size=4, seqlen=1024, device=device, blocking=False, option="dp4")
        #launch_task(model_name="gpt2_large", mode="train", batch_size=4, seqlen=1024, device=device, blocking=False, option="pp4_1")
        #launch_task(model_name="gpt2_large", mode="train", batch_size=4, seqlen=1024, device=device, blocking=False, option="tp4")

        #launch_task(model_name="gpt2_large", mode="train", batch_size=16, seqlen=1024, device=device, blocking=False, option="dp4")
        #launch_task(model_name="gpt2_large", mode="train", batch_size=16, seqlen=1024, device=device, blocking=False, option="pp4_1")
        #launch_task(model_name="gpt2_large", mode="train", batch_size=16, seqlen=1024, device=device, blocking=False, option="tp4")

        #launch_task(model_name="gpt3_xl", mode="train", batch_size=4, seqlen=2048, device=device, blocking=False, option="dp4")
        #launch_task(model_name="gpt3_xl", mode="train", batch_size=4, seqlen=2048, device=device, blocking=False, option="pp4_1")
        #launch_task(model_name="gpt3_xl", mode="train", batch_size=4, seqlen=2048, device=device, blocking=False, option="tp4")
        for p in jobs:
            p.wait()

    """

def main():
    output_file = os.path.join('results', 'prediction', 'carbon_summary.csv')

    # Delete the summary csv file at the beginning if it exists
    if os.path.exists(output_file):
        os.remove(output_file)

    pred()

try:
    main()
except Exception as e:
    print(e)
    for p in jobs:
        print(p.stdout)
        p.kill()