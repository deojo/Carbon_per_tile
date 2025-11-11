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

def launch_task(model_name, mode, batch_size, seqlen, blocking=False, option=""):
    p = subprocess.Popen(
        [
            "python3", "./parse.py",
            "--model_config_path", f"./data/DLmodel_configs/{model_name}.json",
            "--sequence_length", str(seqlen),
            "--batch_size", str(batch_size),
            "--execution_type", mode,
            "--result_dir", "./results",
            "--option", option,
        ], 
        stdout=sys.stdout, 
        stderr=sys.stdout, 
    )

    if blocking:
        p.wait()

    jobs.append(p)

def parse():

    # main
    launch_task(model_name="bert_large", mode="inf", batch_size=8, seqlen=512, blocking=False)
    launch_task(model_name="bert_large", mode="inf", batch_size=16, seqlen=512, blocking=False)
    launch_task(model_name="gpt2_large", mode="inf", batch_size=4, seqlen=1024, blocking=False)
    launch_task(model_name="gpt2_large", mode="inf", batch_size=8, seqlen=1024, blocking=False)
    launch_task(model_name="gpt3_xl", mode="inf", batch_size=2, seqlen=2048, blocking=False)
    launch_task(model_name="gpt3_xl", mode="inf", batch_size=8, seqlen=2048, blocking=False)
    launch_task(model_name="opt_13", mode="inf", batch_size=2, seqlen=2048, blocking=False)
    launch_task(model_name="opt_13", mode="inf", batch_size=8, seqlen=2048, blocking=False)
    launch_task(model_name="gpt3_27", mode="inf", batch_size=2, seqlen=2048, blocking=False)
    launch_task(model_name="gpt3_27", mode="inf", batch_size=8, seqlen=2048, blocking=False)
    launch_task(model_name="switchxl4", mode="inf", batch_size=1, seqlen=512, blocking=False)
    launch_task(model_name="switchxl4", mode="inf", batch_size=2, seqlen=512, blocking=False)

    launch_task(model_name="bert_large", mode="train", batch_size=2, seqlen=512, blocking=False)
    launch_task(model_name="bert_large", mode="train", batch_size=8, seqlen=512, blocking=False)
    launch_task(model_name="gpt2_large", mode="train", batch_size=1, seqlen=1024, blocking=False)
    launch_task(model_name="gpt2_large", mode="train", batch_size=4, seqlen=1024, blocking=False)
    launch_task(model_name="gpt3_xl", mode="train", batch_size=1, seqlen=2048, blocking=False)
    launch_task(model_name="gpt3_xl", mode="train", batch_size=2, seqlen=2048, blocking=False)
    launch_task(model_name="opt_13", mode="train", batch_size=1, seqlen=2048, blocking=False)
    launch_task(model_name="opt_13", mode="train", batch_size=2, seqlen=2048, blocking=False)
    launch_task(model_name="gpt3_27", mode="train", batch_size=1, seqlen=2048, blocking=False)
    launch_task(model_name="gpt3_27", mode="train", batch_size=2, seqlen=2048, blocking=False)
    launch_task(model_name="switchxl4", mode="train", batch_size=1, seqlen=512, blocking=False)
    launch_task(model_name="switchxl4", mode="train", batch_size=2, seqlen=512, blocking=False)
    launch_task(model_name="llama2", mode="train", batch_size=1, seqlen=512, blocking=False)
    launch_task(model_name="llama3", mode="train", batch_size=1, seqlen=512, blocking=False)

    launch_task(model_name="bert_large", mode="inf", batch_size=8, seqlen=512, blocking=False)
    launch_task(model_name="bert_large", mode="inf", batch_size=16, seqlen=512, blocking=False)
    launch_task(model_name="gpt2_large", mode="inf", batch_size=4, seqlen=1024, blocking=False)
    launch_task(model_name="gpt2_large", mode="inf", batch_size=8, seqlen=1024, blocking=False)

    # fusion
    launch_task(model_name="bert_large", mode="inf", batch_size=8, seqlen=512, blocking=False, option="fusion")
    launch_task(model_name="bert_large", mode="inf", batch_size=16, seqlen=512, blocking=False, option="fusion")
    launch_task(model_name="gpt2_large", mode="inf", batch_size=4, seqlen=1024, blocking=False, option="fusion")
    launch_task(model_name="gpt2_large", mode="inf", batch_size=8, seqlen=1024, blocking=False, option="fusion")

    # dist
    launch_task(model_name="gpt2_large", mode="train", batch_size=4, seqlen=1024, blocking=False, option="dp4")
    launch_task(model_name="gpt2_large", mode="train", batch_size=4, seqlen=1024, blocking=False, option="pp4_4")
    launch_task(model_name="gpt2_large", mode="train", batch_size=4, seqlen=1024, blocking=False, option="tp4")

    launch_task(model_name="gpt2_large", mode="train", batch_size=16, seqlen=1024, blocking=False, option="dp4")
    launch_task(model_name="gpt2_large", mode="train", batch_size=16, seqlen=1024, blocking=False, option="pp4_4")
    launch_task(model_name="gpt2_large", mode="train", batch_size=16, seqlen=1024, blocking=False, option="tp4")

    launch_task(model_name="gpt3_xl", mode="train", batch_size=4, seqlen=2048, blocking=False, option="dp4")
    launch_task(model_name="gpt3_xl", mode="train", batch_size=4, seqlen=2048, blocking=False, option="pp4_4")
    launch_task(model_name="gpt3_xl", mode="train", batch_size=4, seqlen=2048, blocking=False, option="tp4")

    for p in jobs:
        p.wait()

def main():
    parse()

try:
    main()
except Exception as e:
    print(e)
    for p in jobs:
        print(p.stdout)
        p.kill()