
import re, pandas as pd

modes = ['op', 'tile']
metrics = ['carbon', 'latency', 'energy_kwh']

def op_matches_prefix(op_list, prefix):
    for entry in op_list:
        if isinstance(entry, list) and len(entry) > 0 and isinstance(entry[0], str):
            match = re.search(f'({")|(".join(i for i in prefix)})', entry[0])
            if match:
                return match.group(0)
            return "Others"
    return "Empty"

def aggregate_carbon(trace):
    #print(f"[DEBUG] trace: {trace}")
    dfs = []
    prefixes = ['VECln', 'VECsoftmax', 'VECadd', 'VEC', 'BMM', 'Linear', 'MEM']

    # Ensure all required columns exist
    for x in ['Fw', 'Bw', 'Acc']:
        for mode in modes:
            for metric in metrics:
                col = f"{x.lower()}_{mode}_{metric}"
                if col not in trace.columns:
                    trace[col] = 0.0
                trace[col] = pd.to_numeric(trace[col], errors="coerce").fillna(0)

    for x in ['Fw', 'Bw', 'Acc']:
        # Assign group label based on op prefixes
        trace['Groups'] = trace[f'{x}Ops'].apply(op_matches_prefix, prefix=prefixes)

        # Group by assigned group label
        grouped = trace.groupby('Groups')

        # Aggregate sum and count
        temp = grouped.agg(
            op_carbon=pd.NamedAgg(column=f"{x.lower()}_op_carbon", aggfunc="sum"),
            op_latency=pd.NamedAgg(column=f"{x.lower()}_op_latency", aggfunc="sum"),
            op_energy_kwh=pd.NamedAgg(column=f"{x.lower()}_op_energy_kwh", aggfunc="sum"),
            tile_carbon=pd.NamedAgg(column=f"{x.lower()}_tile_carbon", aggfunc="sum"),
            tile_latency=pd.NamedAgg(column=f"{x.lower()}_tile_latency", aggfunc="sum"),
            tile_energy_kwh=pd.NamedAgg(column=f"{x.lower()}_tile_energy_kwh", aggfunc="sum"),
            count=pd.NamedAgg(column=f"{x.lower()}_op_carbon", aggfunc="count")
        ).reset_index()

        # Rename columns to indicate pass
        temp = temp.rename(columns={
            'op_carbon': f'op_carbon_{x.lower()}',
            'op_latency': f'op_latency_{x.lower()}',
            'op_energy_kwh': f'op_energy_kwh_{x.lower()}',
            'tile_carbon': f'tile_carbon_{x.lower()}',
            'tile_latency': f'tile_latency_{x.lower()}',
            'tile_energy_kwh': f'tile_energy_kwh_{x.lower()}',
            'count': f'count_{x.lower()}'
        })

        dfs.append(temp)

    # Merge the three aggregated DataFrames on 'Groups'
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = pd.merge(merged_df, df, on='Groups', how='outer')

    # Fill NaNs with 0
    for col in merged_df.columns:
        if col != 'Groups':
            merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce').fillna(0)

    # Compute total sums across passes
    for mode in modes:
        for metric in metrics:                         # now also includes energy_kwh
            merged_df[f'agg_{mode}_{metric}'] = (
                merged_df.get(f'{mode}_{metric}_fw', 0) +
                merged_df.get(f'{mode}_{metric}_bw', 0) +
                merged_df.get(f'{mode}_{metric}_acc', 0)
            )

    merged_df['agg_count'] = (
        merged_df.get('count_fw', 0) +
        merged_df.get('count_bw', 0) +
        merged_df.get('count_acc', 0)
    )

    return merged_df

def replicate_layer(trace, model_name, n_layer, options=""):

    if n_layer == 0:
        return trace

    if "bert" in model_name.lower():
        start = trace['Name'].loc[lambda x: x=="bert_encoder_layer_0_attention_self_query"].index.item()
        end = trace['Name'].loc[lambda x: x=="bert_encoder_layer_0_output_layer_norm"].index.item() + 1
    elif "gpt" in model_name.lower():
        if "transformer_h_0_ln_1_grad" in trace['Name'].values:
            start = trace['Name'].loc[lambda x: x=="transformer_h_0_ln_1_grad"].index
        else:
            start = trace['Name'].loc[lambda x: x=="transformer_h_0_ln_1"].index
        start = start.item()
        end = trace['Name'].loc[lambda x: x=="add_15"].index.item() + 1
    elif "opt" in model_name.lower():
        start = trace['Name'].loc[lambda x: x=="model_decoder_layers_0_self_attn_layer_norm"].index.item()
        end = trace['Name'].loc[lambda x: x=="view_11"].index.item() + 1
    elif "llama" in model_name.lower():
        start = trace['Name'].loc[lambda x: x=="model_layers_0_input_layernorm_weight"].index.item()
        end = trace['Name'].loc[lambda x: x=="model_layers_0_post_attention_layernorm_weight"].index.item() + 1
    elif "switch" in model_name.lower():
        # no need to convert
        return trace
    # elif "megatron" in model_name.lower():
    #     start = trace['Name'].loc[lambda x: x=="language_model_encoder_layers_0_input_layernorm"].index.item()
    #     end = trace['Name'].loc[lambda x: x=="make_viewless_tensor_1"].index.item() + 1
    else:
        assert(0)

    prologue = trace.iloc[:start]
    layer = trace.iloc[start:end]
    epilogue = trace.iloc[end:]

    df = pd.concat([prologue, *([layer]*n_layer), epilogue])
    df = df.reset_index(drop=True)

    return df

def aggregate_gpt(trace, model_name, n_layer):
    trace = replicate_layer(trace, model_name, n_layer)
    # print(trace.columns)
    e2e = trace[f"e2e_op_latency"].sum(axis=0)
    fw = trace[f"fw_op_latency"].sum(axis=0)
    bw = trace[f"bw_op_latency"].sum(axis=0)
    bwall = trace[f"bwall_op_latency"].sum(axis=0)
    acc = trace[f"acc_op_latency"].sum(axis=0)

    return e2e, fw, bw, bwall, acc, aggregate_carbon(trace)

def aggregate_tp(trace, model_name, tp_degree, n_layer):
    # replicate layers
    trace = replicate_layer(trace, model_name, n_layer)
    pred_e2e = trace[f"e2e_op_latency"].sum(axis=0)
    return pred_e2e, aggregate_carbon(trace)

def aggregate_dp(trace, model_name, dp_degree, n_layer):
    # replicate layers
    trace = replicate_layer(trace, model_name, n_layer)

    # acc fw latency
    fw_e2e = trace[f"fw_op_latency"].sum(axis=0)

    rows = []
    for i, r in trace.iterrows():
        r = dict(r)
        r["bw_op_latency"] = r["bwall_op_latency"]
        rows.append(dict(r))
    rows = rows[::-1]

    # sum up compute latency and record communication ops
    compute_latency = 0
    comm_ops = [] # (start_time, latency)
    for r in rows:
        if r["Name"].endswith("_grad"):
            comm_ops.append((compute_latency, r["bw_op_latency"]))
        else:
            compute_latency += r["bw_op_latency"]

    # when does comm ends?
    end_time = 0
    for start_time, lat in comm_ops:
        end_time = max(end_time, start_time) + lat

    # acc fw latency
    bw_e2e = max(end_time, compute_latency)

    pred_e2e = fw_e2e + bw_e2e
    return pred_e2e, aggregate_carbon(trace)

def aggregate_pp(trace, model_name, pp_degree, n_layer, num_micro_batch):

    assert(n_layer % pp_degree == 0)
    assert(num_micro_batch == 1) # only support this for now
    per_device_layer = n_layer // pp_degree

    # single layer latency
    print(f"\ntrace[Name]: {trace['Name'].unique()}\n")

    #Old code
    #start = trace['Name'].loc[lambda x: x=="transformer_h_0_ln_1"].index.item()
    #end = trace['Name'].loc[lambda x: x=="add_15"].index.item() + 1

    model_name = model_name.lower()

    # New code Choose start/end ops based on model type
    if "bert" in model_name:
        start_op = "bert_encoder_layer_0_attention_self_query"
        end_op = "bert_encoder_layer_0_output_layer_norm"
    elif "gpt" in model_name or "opt" in model_name:
        if "transformer_h_0_ln_1" in trace["Name"].values:
            start_op = "transformer_h_0_ln_1"
        elif "transformer_h_0_ln_1_grad" in trace["Name"].values:
            start_op = "transformer_h_0_ln_1_grad"
        else:
            raise ValueError("GPT layer start op not found.")
        end_op = "add_15"
    elif "llama" in model_name:
        start_op = "model_layers_0_input_layernorm"
        end_op = "model_layers_0_post_attention_layernorm"
    elif "switch" in model_name:
        raise NotImplementedError("SwitchTransformer is not supported for pipeline parallelism in this function.")
    else:
        raise NotImplementedError(f"Unsupported model type: {model_name}")

    # Resolve indices
    start_matches = trace['Name'].loc[lambda x: x == start_op]
    end_matches = trace['Name'].loc[lambda x: x == end_op]

    if start_matches.empty or end_matches.empty:
        raise ValueError(f"Start or end op not found in trace. Start: {start_op}, End: {end_op}")

    start = start_matches.index[0]
    end = end_matches.index[0] + 1
    ### 

    begin = trace.iloc[:start]
    layer = trace.iloc[start:end]
    sendrecv = trace.iloc[end:end+1]
    end = trace.iloc[end+1:]

    begin_fw_latency = begin[f"fw_op_latency"].sum(axis=0)
    end_bw_latency = begin[f"bwall_op_latency"].sum(axis=0)

    layer_fw_latency = layer[f"fw_op_latency"].sum(axis=0)
    layer_bw_latency = layer[f"bwall_op_latency"].sum(axis=0)

    per_device_layer_fw_latency = layer_fw_latency * per_device_layer
    per_device_layer_bw_latency = layer_bw_latency * per_device_layer

    sendrecv_fw_latency = sendrecv[f"fw_op_latency"].sum(axis=0)
    sendrecv_bw_latency = sendrecv[f"bwall_op_latency"].sum(axis=0)

    end_fw_latency = end[f"fw_op_latency"].sum(axis=0)
    begin_bw_latency = end[f"bwall_op_latency"].sum(axis=0)

    pred_e2e = 0

    # # move to the last rank device
    # pred_e2e += begin_fw_latency + (per_device_layer_fw_latency + sendrecv_fw_latency*2)*(pp_degree-1)

    # # remaining fw and bw on the last rank device
    # pred_e2e += (per_device_layer_fw_latency + end_fw_latency)*(num_micro_batch) # last rank device does not need to send
    # pred_e2e += (begin_bw_latency + per_device_layer_bw_latency)*(num_micro_batch) # last rank device does not need to send
    # pred_e2e += sendrecv_bw_latency*2 # send to the next device

    # # move to the first rank device
    # pred_e2e += (per_device_layer_bw_latency + sendrecv_bw_latency*2)*(pp_degree-1)
    # pred_e2e -= sendrecv_bw_latency*2 # last bw layer does not need to send
    # pred_e2e += end_bw_latency

    # move to the last rank device
    print(per_device_layer_fw_latency, sendrecv_fw_latency, per_device_layer_bw_latency, sendrecv_bw_latency, sep=", ")
    print(begin_fw_latency, begin_bw_latency, end_fw_latency, end_bw_latency, sep=", ")

    pred_e2e = \
                begin_fw_latency \
              + (per_device_layer_fw_latency + sendrecv_fw_latency*2)*(pp_degree - 1) \
              + (per_device_layer_fw_latency + end_fw_latency + sendrecv_fw_latency)*(num_micro_batch) \
              + (per_device_layer_bw_latency + begin_bw_latency + sendrecv_bw_latency)*(num_micro_batch) \
              + (per_device_layer_bw_latency + sendrecv_bw_latency)*(pp_degree - 1) \
              + end_bw_latency \
        # + end_fw_latency*(num_micro_batch) \
        # + begin_bw_latency*(num_micro_batch) \

    return pred_e2e, aggregate_carbon(trace)

def aggregate_latency(
        df, 
        model_name, 
        distributed,
        dp_degree,
        pp_degree,
        pp_num_microbatch,
        tp_degree,
        fusion,
        n_layer,
):

    fw = 0
    bw = 0
    bwall = 0
    acc = 0

    if distributed:
        if dp_degree > 1:
            e2e, carb = aggregate_dp(df, model_name, dp_degree, n_layer)
        elif tp_degree > 1:
            e2e, carb = aggregate_tp(df, model_name, tp_degree, n_layer)
        elif pp_degree > 1:
            e2e, carb = aggregate_pp(df, model_name, pp_degree, n_layer, pp_num_microbatch)
    elif fusion:
        e2e, fw, bw, bwall, acc, carb = aggregate_gpt(df, model_name, 0)
    else:
        e2e, fw, bw, bwall, acc, carb = aggregate_gpt(df, model_name, n_layer)

    return e2e, fw, bw, bwall, acc, carb