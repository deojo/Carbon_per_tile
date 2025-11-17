from pathlib import Path
import pandas as pd
import json
import ast
import torch
import numpy as np
import os
from ..Model.model_provider import model_provider
from ..Tracing.parse import parse_trace
from ..Tracing.trace import trace_graph
from .aggregator import aggregate_latency
from .memory_model import AdaptiveMemoryCommModel
from ..Model.mlp_wave_vec import MLPWaveVec
from ..Model.mlp_wave_mm import MLPWaveMM

ops_dict = {
    "add" : 1.,
    "addu": 1.,
    "mul" : 1.,
    "mulu": 1.,
    "pow" : 1.,
    "powu": 1.,
    "div" : 1.,
    "divu": 1.,
    "tanh": 1.,
    "ln"  : 6., # mean, var, sum, div, sqrt, acc
    "softmax" : 5.,
    "relu" : 1.,
    "gelu" : 1.,
    "silu" : 1.,
    "MEM" : 0.,
}

modes = ['op', 'wave', 'tile']

def ensure_sequence(val):
    return val if isinstance(val, (list, tuple, np.ndarray)) else [val]

def safe_get(item, idx, default=0.0):
    try:
        return item[idx]
    except (IndexError, TypeError):
        return default

def safe_tensor_to_list(tensor):
    if isinstance(tensor, (list, tuple)):
        return list(tensor)
    elif hasattr(tensor, 'tolist'):
        return tensor.tolist()
    return []

def reduce_mul(myList):
    # Multiply elements one by one
    result = 1
    for x in myList:
        result = result * x
    return result

def dump_df(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

class MLPredictor:
    def __init__(self, predictor_path, meta_table_path):
        config_path = predictor_path/"config.json"

        self.model = model_provider(config_path)

        if os.path.isfile(predictor_path/"model.pth"):
            self.model.load_state(load_path=predictor_path/"model.pth")

        if hasattr(self.model, "set_meta_table") and meta_table_path != "":
            self.model.set_meta_table(meta_table_path)
        
    def predict(self, opname, kernel_arguments, device_config):

        all_params = kernel_arguments
        all_params.update(device_config)

        inputs = []
        for kw in self.model.features:
            inputs.append(all_params[kw])
        inputs = torch.tensor(inputs).float()
        inputs = inputs.reshape(1, -1)

        # to device
        inputs = inputs.to(self.model.device)
        self.model = self.model.to(self.model.device)

        # predict runtime with model
        culib = "cu121"
        pred = self.model(opname=opname, x=inputs, device=all_params["Device"], culib=culib)
        
        for i, p in enumerate(pred):
            p = p.detach().cpu()             
            if p.numel() == 1:
                p = np.maximum(p, 0)
                pred[i] = float(p.item())
            else:
                pred[i] = p
            #pred[i] = float(p.squeeze().item())  # convert to float


        #print(f"\npred: {pred}\n")
        return pred

class OperatorPredictor():
    def __init__(
        self, 
        predictor_path, 
        tile_dataset_dir,
    ):

        if str(tile_dataset_dir) == "":
            linear_tile_dataset = ""
            bmm_tile_dataset = ""
            vec_tile_dataset = ""
        else:
            linear_tile_dataset = tile_dataset_dir/"linear.csv"
            bmm_tile_dataset = tile_dataset_dir/"bmm.csv"
            vec_tile_dataset = tile_dataset_dir/"vec.csv"

        predictor_path = Path(predictor_path)

        self.linear_predictor = MLPredictor(predictor_path/"LINEAR", meta_table_path=linear_tile_dataset)
        self.bmm_predictor = MLPredictor(predictor_path/"BMM", meta_table_path=bmm_tile_dataset)
        self.vec_predictor = MLPredictor(predictor_path/"VEC", meta_table_path=vec_tile_dataset)
        self.softmax_predictor = MLPredictor(predictor_path/"SOFTMAX", meta_table_path=vec_tile_dataset)
        self.ln_predictor = MLPredictor(predictor_path/"LN", meta_table_path=vec_tile_dataset)
        # Adaptive memory and interconnect model
        self.mem_comm_model = AdaptiveMemoryCommModel()

    def predict_phase(
                self,
                opname,
                device_config,
                input_shapes,
                output_shape,
                ops,
    ):
        latency = 0.0
        mem_effective_bytes = 0.0
        pr = [0.0, float("NaN"), float("NaN"), float("NaN"), float("NaN")]

        # count mem
        num_input_elem = [reduce_mul(s) for s in input_shapes]
        num_input_elem = sum(num_input_elem) 
        num_output_elem = reduce_mul(output_shape)

        if opname == "fused":
            assert(input_shapes is not None)
            assert(output_shape is not None)

            if ops == []:
                return 0

            # find representative vec op
            rep_op = ops[-1]

            # count mem
            #num_input_elem = [reduce_mul(s) for s in input_shapes]
            #num_input_elem = sum(num_input_elem) 
            #num_output_elem = reduce_mul(output_shape)
            # num_inter_elem = 0
            # for op in ops:
            #     opname, args = op
            #     if opname == "ln":
            #         B, H = args
            #         num_inter_elem += B*H*2

            memPerO = (num_input_elem + num_output_elem) * 4 / num_output_elem

            # count ops
            acc_ops = 0.

            for op in ops:
                opname, args = op
                opname = opname.replace("VEC", "")
                if opname == "MEM":
                    ops = 0.
                else:
                    opsPerO = ops_dict[opname]
                    ops = opsPerO * args[0] * args[1]
                acc_ops += ops
            opsPerO = acc_ops / num_output_elem

            # call predictor

            opname, args = rep_op
            if opname.startswith("VEC"):
                opname = opname.replace("VEC", "")
                B, H = args
                if "softmax" in opname.lower():
                    pr = self.softmax_predictor.predict(opname=[opname],kernel_arguments={"B":B,"H":H, "MemPerO":memPerO, "OpsPerO":opsPerO},device_config=device_config)
                    # print("last bw util", self.softmax_predictor.model.last_bw_util)
                
                elif "ln" in opname.lower():
                    pr = self.ln_predictor.predict(opname=[opname],kernel_arguments={"B":B,"H":H, "MemPerO":memPerO, "OpsPerO":opsPerO},device_config=device_config)
                    # print("last bw util", self.ln_predictor.model.last_bw_util)
                
                else:
                    pr = self.vec_predictor.predict(opname=[opname],kernel_arguments={"B":B,"H":H, "MemPerO":memPerO, "OpsPerO":opsPerO},device_config=device_config)
                    # print("last bw util", self.vec_predictor.model.last_bw_util)

                # Apply adaptive memory scaling to account for L2/overlap on memory-bound vector ops
                total_bytes = float((num_input_elem + num_output_elem) * 4)
                ratio = self.mem_comm_model.memory_scaling_ratio(
                    bytes_total=total_bytes,
                    device_config=device_config,
                    access_pattern="streaming",
                )
                latency += pr[0] * ratio
                
            elif opname == "MEM":
                # Streaming memory copy modeled via adaptive HBM
                total_bytes = float((num_input_elem + num_output_elem) * 4)
                mem_lat_s, eff_bytes, _ = self.mem_comm_model.compute_mem_latency_bytes(
                    bytes_total=total_bytes,
                    device_config=device_config,
                    access_pattern="streaming",
                )
                latency += mem_lat_s
                mem_effective_bytes += eff_bytes
            elif opname == "misc":
                assert(0)
            else:
                raise NotImplementedError

        else:
            for op in ops:
                opname, args = op
                
                if opname == "Linear":
                    B = 1
                    M, N, K = args
                    pr = self.linear_predictor.predict(opname=["linear"],kernel_arguments={"B":B,"M":M,"N":N,"K":K},device_config=device_config)
                    total_bytes = float((num_input_elem + num_output_elem) * 4)
                    ratio = self.mem_comm_model.memory_scaling_ratio(
                        bytes_total=total_bytes,
                        device_config=device_config,
                        access_pattern="streaming",
                    )
                    latency += pr[0] * ratio
                    
                elif opname == "BMM":
                    B, M, N, K = args
                    pr = self.bmm_predictor.predict(opname=["bmm"],kernel_arguments={"B":B,"M":M,"N":N,"K":K},device_config=device_config)
                    latency += pr[0]
                    
                elif opname.startswith("VEC"):
                    assert(input_shapes is not None)
                    assert(output_shape is not None)
                    opname = opname.replace("VEC", "")
                    B, H = args
                    num_input_elem = [reduce_mul(s) for s in input_shapes]
                    num_input_elem = sum(num_input_elem) 
                    num_output_elem = reduce_mul(output_shape)
                    memPerO = (num_input_elem + num_output_elem) * 4 / num_output_elem
                    opsPerO = ops_dict[opname]

                    if "softmax" in opname.lower():
                        pr = self.softmax_predictor.predict(opname=[opname],kernel_arguments={"B":B,"H":H, "MemPerO":memPerO, "OpsPerO":opsPerO},device_config=device_config)    
                     
                    elif "ln" in opname.lower():
                        pr = self.ln_predictor.predict(opname=[opname],kernel_arguments={"B":B,"H":H, "MemPerO":memPerO, "OpsPerO":opsPerO},device_config=device_config)
        
                    else:
                        pr = self.vec_predictor.predict(opname=[opname],kernel_arguments={"B":B,"H":H, "MemPerO":memPerO, "OpsPerO":opsPerO},device_config=device_config)
                        
                    latency += pr[0]

                      
                elif opname == "MEM":
                    total_bytes = 0.0
                    for shape in args:
                        total_bytes += float(reduce_mul(shape) * 4)
                    mem_lat_s, eff_bytes, _ = self.mem_comm_model.compute_mem_latency_bytes(
                        bytes_total=total_bytes,
                        device_config=device_config,
                        access_pattern="streaming",
                    )
                    latency += mem_lat_s
                    mem_effective_bytes += eff_bytes
                
                elif opname == "ALLREDUCE" or opname=="ALLREDUCE_ASYNC":
                    buffer_size = args[0]
                    num_gpu = 8
                    latency += self.mem_comm_model.compute_allreduce_latency_ring(
                        bytes_total=float(buffer_size * 4),
                        link_bw_gbps=float(self.link_bw),
                        num_gpu=num_gpu,
                    )
                    # latency += (buffer_size * 4) * self.bw_coeff + self.bw_bias

                elif opname == "SENDRECV":
                    buffer_size = args[0]
                    latency += self.mem_comm_model.compute_sendrecv_latency(
                        bytes_total=float(buffer_size * 4),
                        link_bw_gbps=float(self.link_bw),
                    )
                    # latency += (buffer_size * 4) * self.bw_coeff + self.bw_bias
                
                elif opname == "misc":
                    assert(0)
                
                else:
                    print(opname)
                    raise NotImplementedError

        # Convert effective HBM bytes to energy for carbon accounting
        mem_energy_kwh = self.mem_comm_model.bytes_to_kwh(mem_effective_bytes)
        return (latency, pr[1], pr[2], pr[3], pr[4], mem_energy_kwh)
        #return (latency, pr[1], pr[2], pr[3], pr[4])


    def predict(
                self,
                device_config,
                x,
    ):

        opname = x.loc["OpName"] 
        input_shapes = x.loc["InputShapes"]
        output_shape = x.loc["OutputShape"]
        device_name = device_config['Device']

        self.mem_bw = float(device_config["Mem_Bw"]) # gb/s

        self.util = 0.72
        if device_name == "NVIDIA H100 80GB HBM3":
            self.link_bw = 900 * self.util / 2 # in GB/s
        elif device_name == "Tesla V100-SXM2-16GB":
            self.link_bw = 300 * self.util / 2
        elif device_name == "NVIDIA A100-PCIE-40GB" or device_name == "NVIDIA A100-SXM4-40GB":
            self.link_bw = 600 * self.util / 2 # in GB/s
        else:
            self.link_bw = None

        # for habitat
        if hasattr(self.vec_predictor.model, "meta_table"):
            habitat_vec_ref_device="Tesla V100-PCIE-32GB" if device_name != "Tesla V100-PCIE-32GB" else "Tesla P100-PCIE-16GB"
            self.vec_predictor.model.meta_table.set_device(habitat_vec_ref_device)
            self.ln_predictor.model.meta_table.set_device(habitat_vec_ref_device)
            self.softmax_predictor.model.meta_table.set_device(habitat_vec_ref_device)

        fw_ops = x.loc["FwOps"]
        bw_ops = x.loc["BwOps"]
        acc_ops = x.loc["AccOps"]
        
        fw_out = self.predict_phase(
            device_config=device_config, input_shapes=input_shapes,
            output_shape=output_shape, ops=fw_ops, opname=opname)
        bw_out = self.predict_phase(
            device_config=device_config, input_shapes=input_shapes,
            output_shape=output_shape, ops=bw_ops, opname=opname)
        acc_out = self.predict_phase(
            device_config=device_config, input_shapes=input_shapes,
            output_shape=output_shape, ops=acc_ops, opname=opname)

        if not isinstance(fw_out, (list, tuple, np.ndarray)):
            fw_out = [fw_out]
        if not isinstance(bw_out, (list, tuple, np.ndarray)):
            bw_out = [bw_out]
        if not isinstance(acc_out, (list, tuple, np.ndarray)):
                acc_out = [acc_out]

        return (
            safe_get(fw_out, 0) * 1000, safe_get(bw_out, 0) * 1000, safe_get(acc_out, 0) * 1000,  # op latency (ms)
            safe_get(fw_out, 1) * 1000, safe_get(bw_out, 1) * 1000, safe_get(acc_out, 1) * 1000,  # wave latency (ms)
            safe_get(fw_out, 2) * 1000, safe_get(bw_out, 2) * 1000, safe_get(acc_out, 2) * 1000,  # tile latency (ms)
            safe_get(fw_out, 3, 1.0), safe_get(bw_out, 3, 1.0), safe_get(acc_out, 3, 1.0),        # util (default 1.0)
            safe_tensor_to_list(safe_get(fw_out, 4, [])),
            safe_tensor_to_list(safe_get(bw_out, 4, [])),
            safe_tensor_to_list(safe_get(acc_out, 4, [])),
            safe_get(fw_out, 5), safe_get(bw_out, 5), safe_get(acc_out, 5),
        )
            

class NeusightPredictor():
    def __init__(
        self,
        predictor_name,
        predictor_path,
        tile_dataset_dir,
    ):

        self.name = predictor_name
        self.CI_US = 380
        if tile_dataset_dir != "":
            tile_dataset_dir = Path(tile_dataset_dir)

        self.predictor = OperatorPredictor(
                    predictor_path=predictor_path, 
                    tile_dataset_dir=tile_dataset_dir,
                )

    def predict(self, 
                device_config_path, # hardware description
                model_config_path, # model configuration
                sequence_length,
                batch_size,
                result_dir,
                model_name=None,
                execution_type="inf",
                options="", # additional options
            ):
        
        result_dir = Path(result_dir)

        is_train = (execution_type == "train")
        
        if model_name is None:
            model_name = Path(model_config_path).name.split(".")[0]

        # parse options
        fusion = False
        dp_degree = 1
        tp_degree = 1
        pp_degree = 1
        pp_num_microbatch = 1
        distributed = False
        single_layer = True

        import re

        if options == "":
            pass
        elif options == "fusion":
            fusion = True
            single_layer = False
        elif re.match(r"dp\d+", options):
            distributed = True
            dp_degree = int(options[2:])
        elif re.match(r"tp\d+", options):
            distributed = True
            tp_degree = int(options[2:])
        elif re.match(r"pp\d+_\d+", options):
            distributed = True
            pp_degree = int(options[2:].split("_")[0])
            pp_num_microbatch = int(options[2:].split("_")[1])
        else:
            assert(0)

        if "switch" in model_name or fusion:
            single_layer = False


        if fusion:
            model_tag = f"{model_name}-{execution_type}-{sequence_length}-{batch_size}-fusion"
            trace_name = result_dir/f"opgraph_raw/{model_name}-{execution_type}-{sequence_length}-{batch_size}-fusion.csv"
            parse_name = result_dir/f"opgraph/{model_name}-{execution_type}-{sequence_length}-{batch_size}-fusion.csv"
        elif distributed:
            model_tag = f"{model_name}-{execution_type}-{sequence_length}-{batch_size}-{options}"
            if dp_degree > 1:
                trace_name = result_dir/f"opgraph_raw/{model_name}-{execution_type}-{sequence_length}-{batch_size // dp_degree}.csv"
                parse_name = result_dir/f"opgraph/{model_name}-{execution_type}-{sequence_length}-{batch_size}-dp{dp_degree}.csv"
            elif tp_degree > 1:
                trace_name = result_dir/f"opgraph_raw/{model_name}-{execution_type}-{sequence_length}-{batch_size}.csv"
                parse_name = result_dir/f"opgraph/{model_name}-{execution_type}-{sequence_length}-{batch_size}-tp{tp_degree}.csv"
            elif pp_degree > 1:
                trace_name = result_dir/f"opgraph_raw/{model_name}-{execution_type}-{sequence_length}-{batch_size // pp_num_microbatch}.csv"
                parse_name = result_dir/f"opgraph/{model_name}-{execution_type}-{sequence_length}-{batch_size}-pp{pp_degree}_{pp_num_microbatch}.csv"
        else:
            model_tag = f"{model_name}-{execution_type}-{sequence_length}-{batch_size}"
            trace_name = result_dir/f"opgraph_raw/{model_tag}.csv"
            parse_name = result_dir/f"opgraph/{model_tag}.csv"

        device_config_path = Path(device_config_path)
        device_config_path = device_config_path.absolute()

        print(f"device_config_path: {device_config_path}")
        with open(device_config_path, "r") as f:
            device_config = json.load(f)

        # trace raw operator graph
        print(trace_name)
        if os.path.exists(trace_name):
            print("already exists : ", os.path.realpath(trace_name))
            pass
        else:
            df, _ = trace_graph(
                                model_config_path=model_config_path, 
                                sequence_length=sequence_length, 
                                batch_size=batch_size, 
                                is_train=is_train, 
                                bench=False, 
                                single_layer=single_layer, 
                                fusion=fusion,
                                distributed=distributed,
                                dp_degree=dp_degree,
                                pp_degree=pp_degree,
                                pp_num_microbatch=pp_num_microbatch,
                                tp_degree=tp_degree,
                            )
            dump_df(df, trace_name)

        # parse operator graph
        print(parse_name)
        if os.path.exists(parse_name):
            print("already exists : ", os.path.realpath(parse_name))
            
        else:
            df = parse_trace(
                    trace_name, 
                    is_train=is_train, 
                    bench=False, 
                    fusion=fusion,
                    distributed=distributed,
                    dp_degree=dp_degree,
                    pp_degree=pp_degree,
                    pp_num_microbatch=pp_num_microbatch,
                    tp_degree=tp_degree,
                )
            dump_df(df, parse_name)

        # annotate operator graph with prediction
        df = pd.read_csv(parse_name, converters={"FwOps": ast.literal_eval,
                                                "BwOps": ast.literal_eval,
                                                "AccOps": ast.literal_eval,
                                                "InputShapes": ast.literal_eval,
                                                "OutputShape": ast.literal_eval}
                                                )
        
        df[
            ["fw_op_latency", "bw_op_latency", "acc_op_latency",
            "fw_wave_latency", "bw_wave_latency", "acc_wave_latency",
            "fw_tile_latency", "bw_tile_latency", "acc_tile_latency",
            "fw_util", "bw_util", "acc_util",
            "fw_tile_dim", "bw_tile_dim", "acc_tile_dim",
            "fw_mem_kwh", "bw_mem_kwh", "acc_mem_kwh"]
        ] = df.apply(lambda x: pd.Series(self.predictor.predict(device_config, x)), axis=1)


        for lat in modes:
            df[f"bwall_{lat}_latency"] = (
                df[f"bw_{lat}_latency"].fillna(0.0) + df[f"acc_{lat}_latency"].fillna(0.0)
            )
            df[f"e2e_{lat}_latency"] = (
                df[f"fw_{lat}_latency"].fillna(0.0) +
                df[f"bw_{lat}_latency"].fillna(0.0) +
                df[f"acc_{lat}_latency"].fillna(0.0)
            )



        carbon_input_cols = ["FwOps", "BwOps", "AccOps",
            "fw_op_latency", "bw_op_latency", "acc_op_latency",
            "fw_tile_latency", "bw_tile_latency", "acc_tile_latency",
            "e2e_op_latency", "e2e_tile_latency",
            "fw_util", "bw_util", "acc_util"
        ]

        device = device_config['Device'].replace(' ', '_')
        #print(f"Using device: {device}")

        # Compute carbon per row and add new columns to df
        carbon_dicts = df.apply(
            lambda row: self.compute_total_carbon(device, row[carbon_input_cols].to_dict(), distributed),
            axis=1
        )

        # Expand the returned dictionaries into new DataFrame columns and join to original df
        carbon_df = pd.DataFrame(carbon_dicts.tolist())
        df = pd.concat([df, carbon_df], axis=1)


        for col in ["fw_op_carbon", "bw_op_carbon", "acc_op_carbon"]:
            if col not in df:
                df[col] = 0.0

        # Define correct masks
                        
        df.fillna("0.0e0", inplace=True)
        with open(model_config_path) as f:
            config_json = json.load(f)
        if "gpt" in model_name:
            n_layer = config_json["n_layer"]
        elif "switch" in model_name:
            n_layer = config_json["num_layers"]
        else:
            n_layer = config_json["num_hidden_layers"]

        # accumulate latency

        out_path = result_dir/f"prediction/{device_config['Device'].replace(' ', '_')}/{self.name}/{model_tag}.csv"
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)

        
        # accumulate latency + carbon + energy
        e2e, fw, bw, bwall, acc, carb = aggregate_latency(
                                                    df, 
                                                    model_name, 
                                                    distributed=distributed,
                                                    dp_degree=dp_degree,
                                                    pp_degree=pp_degree,
                                                    pp_num_microbatch=pp_num_microbatch,
                                                    tp_degree=tp_degree,
                                                    fusion=fusion,
                                                    n_layer=n_layer,
                                                )

        # Remove the row with Groups == "Empty"
        carb = carb[carb['Groups'] != "Empty"]
    
        print(f'agg_tile_energy_kwh: {carb[["Groups", "agg_tile_energy_kwh", "agg_count"]]}')
        # Compute totals
        total_op_carbon = carb["agg_op_carbon"].sum()
        total_op_latency = carb["agg_op_latency"].sum()
        total_op_energy_kwh = carb["agg_op_energy_kwh"].sum()

        total_tile_carbon = carb["agg_tile_carbon"].sum()
        total_tile_latency = carb["agg_tile_latency"].sum()
        total_tile_energy_kwh = carb["agg_tile_energy_kwh"].sum()
        print(f"total_tile_energy_kwh: {total_tile_energy_kwh}")

        total_count = carb["agg_count"].sum()

        # Compute percentages
        carb["pct_op_carbon"] = (carb["agg_op_carbon"] / total_op_carbon * 100).round(2)
        carb["pct_op_latency"] = (carb["agg_op_latency"] / total_op_latency * 100).round(2)
        carb["pct_op_energy_kwh"] = (carb["agg_op_energy_kwh"] / total_op_energy_kwh * 100).round(2)

        # Add averages per op
        for mode in ['op', 'tile']:
            for metric in ['carbon', 'latency', 'energy_kwh']:
                carb[f"avg_{mode}_{metric}"] = (
                    carb[f"agg_{mode}_{metric}"] / carb["agg_count"]
                ).replace([np.inf, -np.inf], 0).fillna(0).round(15)

        # Create a summary row
        summary_row = pd.DataFrame([{
            "Groups": "TOTAL",
            "agg_op_carbon": total_op_carbon,
            "agg_op_latency": total_op_latency,
            "agg_op_energy_kwh": total_op_energy_kwh,
            "agg_tile_carbon": total_tile_carbon,
            "agg_tile_latency": total_tile_latency,
            "agg_tile_energy_kwh": total_tile_energy_kwh,
            "agg_count": total_count,
            "pct_op_carbon": 100.0,
            "pct_op_latency": 100.0,
            "pct_op_energy_kwh": 100.0
        }])

        # Append the summary row to the original DataFrame
        carb = pd.concat([carb, summary_row], ignore_index=True)

        #print(f"\nNew value carb: \n{carb}\n")

        # Select specific columns (now including energy_kwh)
        selected_columns = carb[
            [
                'Groups',
                'agg_op_carbon', 'agg_op_latency', 'agg_op_energy_kwh',
                'avg_op_carbon', 'avg_op_latency', 'avg_op_energy_kwh',
                'avg_tile_carbon', 'avg_tile_latency', 'avg_tile_energy_kwh',
                'agg_count',
                'pct_op_carbon', 'pct_op_latency', 'pct_op_energy_kwh'
            ]
        ]

    
        # Target file
        output_file = os.path.join(result_dir, 'prediction', 'carbon_summary.csv')
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Append to the CSV file
        #write_header = not os.path.exists(output_file)
        #with open(output_file, mode='a') as f:
        #    f.write(f"\n{device} / {model_tag}\n")  
        #    selected_columns.to_csv(f, header=True, index=False)

        write_header = not os.path.exists(output_file)

        # Open the file in append mode
        with open(output_file, mode='a') as f:
            line = f"\n{device} / {model_tag}\n"
            f.write(line)        # write header to file
            print(line, end='')  # print header to console

            # Write the DataFrame to file
            selected_columns.to_csv(f, header=True, index=False)
            # Print the DataFrame to console
            print(selected_columns.to_string(index=False))
    
        
        print(f"E2E latency for {model_tag} on {device_config_path.name}:", round(e2e, 2), "ms")
        print("\n")

        json_dict = {
            "num_layer": n_layer,
            "e2e_latency": float(e2e),
            "fw_latency": float(fw),
            "bwall_latency": float(bwall),
            "bw_latency": float(bw),
            "acc_latency": float(acc),
        }

        # Save the dictionary to a JSON file
        with open(out_path.with_suffix(".json"), 'w') as file:
            json.dump(json_dict, file, indent=4)


    def compute_total_carbon(self, device: str, latency_data: dict, distributed: bool):
        DEVICE_TDP = {
            "NVIDIA_A100-SXM4-40GB": 400.0,
            "Tesla_P100-PCIE-16GB": 250.0,
            "Tesla_V100-PCIE-32GB": 250.0,
            "Tesla_V100-SXM2-16GB": 300.0,
            "NVIDIA_A100_80GB_PCIe": 300.0,
            "NVIDIA_H100_80GB_HBM3": 700.0,
            "Tesla_P4": 75.0,
            "NVIDIA_A100-PCIE-40GB": 250.0,
            "NVIDIA_L4": 72.0,
            "Tesla_T4": 70.0,
            "NVIDIA_B200": 1200.0,
        }

        EMBODIED_CARBON_TOTAL = {
            "Tesla_V100-PCIE-32GB": 163_793.98,
            "Tesla_V100-SXM2-16GB": 163_793.98,
            "NVIDIA_A100_80GB_PCIe": 150_000.00,
            "NVIDIA_A100-PCIE-40GB": 24_666.72,
            "NVIDIA_A100-SXM4-40GB": 24_666.72,
            "Tesla_P100-PCIE-16GB": 14_000.00,
            "NVIDIA_H100_80GB_HBM3": 164_000,
            "Tesla_P4": 10_300.00,
            "NVIDIA_L4": 12_000.00,
            "Tesla_T4": 10_300.00,
            "NVIDIA_B200": 63_700.00 * 5.0606,
        }

        EMBODIED_CARBON_MEM = {
            "NVIDIA_A100_80GB_PCIe": 80_000.00,
            "NVIDIA_A100-PCIE-40GB": 40_000.00,
            "NVIDIA_A100-SXM4-40GB": 40_000.00,
            "NVIDIA_H100_80GB_HBM3": 68_250,
            "NVIDIA_B200": 186_000.00,
        }

        num_gpu = 1
        if distributed:
            print(f"\nReceived distributed for device: {device}\n")
            num_gpu = 8

        # Constants
        CI_US = 380  # gCO₂ per kWh
        LT_MS = 6 * 365 * 24 * 60 * 60 * 1000  # 6 years in ms

        expected_latency_cols = [
            "fw_op_latency", "bw_op_latency", "acc_op_latency",
            "fw_tile_latency", "bw_tile_latency", "acc_tile_latency"
        ]

        expected_keys = [
            "fw_op_carbon", "bw_op_carbon", "acc_op_carbon",
            "fw_tile_carbon", "bw_tile_carbon", "acc_tile_carbon",
            "e2e_op_carbon", "e2e_tile_carbon"
        ]

        tdp = DEVICE_TDP.get(device)
        carbon_per_latency = {}

        for col_name in expected_latency_cols:
            embodied_carbon = EMBODIED_CARBON_TOTAL.get(device)
            latency_ms = latency_data.get(col_name, 0.0)
            if pd.isna(latency_ms):
                latency_ms = 0.0

            op_pass = ''
            if "fw" in col_name:
                op_pass = 'fw'
            elif "bw" in col_name:
                op_pass = 'bw'
            elif "acc" in col_name:
                op_pass = 'acc'

            util = latency_data.get(f"{op_pass}_util", 1.0)
            if pd.isna(util):
                util = 1.0

            ops = latency_data.get(f"{op_pass.upper()}Ops", [])
            latency_hr = latency_ms / (1000 * 3600)  # ms → hr

            # Energy consumption (kWh)
            if ops and any(op[0] == "MEM" for op in ops):
                energy_kWh = latency_data.get(f'{op_pass}_mem_kwh', 0.0)
                print(f"\n energy_kwh for MEM: {energy_kWh}\n")
                embodied_carbon = EMBODIED_CARBON_MEM.get(device)
            else:
                energy_kWh = (tdp * num_gpu * latency_hr * util) / 1000

            # Carbon emissions
            carbon_op = energy_kWh * CI_US
            carbon_em = (latency_ms / LT_MS) * embodied_carbon * num_gpu
            carbon_total = carbon_op + carbon_em

            # Save both carbon and energy
            carbon_key = col_name.replace('latency', 'carbon')
            energy_key = col_name.replace('latency', 'energy_kwh')

            carbon_per_latency[carbon_key] = f"{carbon_total:.6e}"
            carbon_per_latency[energy_key] = f"{energy_kWh:.6e}"

        # Fill missing expected keys
        for key in expected_keys:
            if key not in carbon_per_latency:
                carbon_per_latency[key] = 0.0

        # Aggregate totals
        carbon_per_latency["e2e_op_carbon"] = (
            float(carbon_per_latency["fw_op_carbon"]) +
            float(carbon_per_latency["bw_op_carbon"]) +
            float(carbon_per_latency["acc_op_carbon"])
        )
        carbon_per_latency["e2e_tile_carbon"] = (
            float(carbon_per_latency["fw_tile_carbon"]) +
            float(carbon_per_latency["bw_tile_carbon"]) +
            float(carbon_per_latency["acc_tile_carbon"])
        )

        #print(f"carbon_per_latency: {carbon_per_latency}")
        return carbon_per_latency



    def ms_to_hours(self, ms):
        return ms / (1000 * 60 * 60)
    
    def safe_float(val, default=1.0):
        try:
            return float(val)
        except (ValueError, TypeError):
            return default
   
   
    
    
