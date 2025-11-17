import argparse
from pathlib import Path
import os
import sys

# Ensure the local NeuSight package is importable before any site-packages
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import neusight

parser = argparse.ArgumentParser()

parser.add_argument("--predictor_name", type=str, help="Name of the predictor")
parser.add_argument("--predictor_path", type=str, help="Path to the predictor")

parser.add_argument("--device_config_path", type=str, help="Path to the device config")
parser.add_argument("--model_config_path", type=str, help="Path to the model config")
parser.add_argument("--sequence_length", type=int, help="Sequence length")
parser.add_argument("--batch_size", type=int, help="Batch size")
parser.add_argument("--execution_type", type=str, help="Execution type")

parser.add_argument("--tile_dataset_dir", type=str, help="Path to the tile dataset directory")
parser.add_argument("--result_dir", type=str, help="Path to the result directory")

parser.add_argument("--options", type=str, help="Options", default="")

parser.add_argument("--running_device", type=str, help="Options", default=None)

args = parser.parse_args()
if args.running_device is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.running_device.split(":")[1]

# initialize neusight predictor
neusight_predictor = neusight.NeusightPredictor(
    predictor_name=args.predictor_name,
    predictor_path=args.predictor_path,
    tile_dataset_dir=args.tile_dataset_dir,
)

# make prediction
neusight_predictor.predict(
    device_config_path=args.device_config_path,
    model_config_path=args.model_config_path,
    sequence_length=args.sequence_length,
    batch_size=args.batch_size,
    execution_type=args.execution_type,
    result_dir=args.result_dir,
    options=args.options,
)
