import pandas as pd
import ast
from ..Opgraph.fuse import fuse_parse

def multiplyList(myList):
    # Multiply elements one by one
    result = 1
    for x in myList:
        result = result * x
    return result

def read_vec_shape(input_shapes, output_shape):
    tot = multiplyList(output_shape)
    H = output_shape[-1]
    B = tot // H

    return B, H

def parse_ops(name, input_shapes, output_shape, meta, vocab_size, contiguous, is_train):
    name = name.lower()

    # for readability
    for i, s in enumerate(input_shapes):
        input_shapes[i] = tuple(s)

    fw_ops = []
    bw_ops = []
    acc_ops = []

    if "norm_weight" in name:
        opname = "Weight"
        acc_ops.append(("VECadd", [1, output_shape[0]]))

    elif ("addmm" in name) or ("nn.modules.linear.Linear" in meta) \
            or ("matmul" in name and "Linear" in meta) \
            or ("matmul_4" in name):
        opname = "Linear"
        if len(input_shapes) < 3:
            op1 = input_shapes[0]
        else:
            bias, op1, op2 = input_shapes
        out = output_shape
        
        B = multiplyList(op1) // op1[-1]
        I = op1[-1]
        O = out[-1]

        fw_ops.append(("Linear", (B, I, O)))
        bw_ops.append(("Linear", (B, O, I)))
        bw_ops.append(("Linear", (O, B, I)))
        acc_ops.append(("VECadd", [1, O*I]))
        acc_ops.append(("VECadd", [1, O]))

    elif "matmul" in name or "bmm" in name:

        opname = "BMM"
        if len(input_shapes) == 2:
            op1, op2 = input_shapes

            M = op1[-2]
            N = op1[-1]
            B = multiplyList(op1) // (M*N)
            K = op2[-1]

            fw_ops.append(("BMM", (B, M, N, K)))
            bw_ops.append(("BMM", (B, M, K, N)))
            bw_ops.append(("BMM", (B, N, M, K)))
        elif len(input_shapes) == 3:
            op0, op1, op2 = input_shapes

            M = op1[-2]
            N = op1[-1]
            B = multiplyList(op1) // (M*N)
            K = op2[-1]

            fw_ops.append(("BMM", (B, M, N, K)))
            fw_ops.append(("VECadd", (B, M*K)))
            bw_ops.append(("BMM", (B, M, K, N)))
            bw_ops.append(("BMM", (B, N, M, K))) 
        else:
            assert(0)

    elif "add" in name:
        opname = "VECadd"
        if len(input_shapes) == 1:
            opname += "u"
        B, H = read_vec_shape(input_shapes, output_shape)
        fw_ops.append((opname, (B, H)))
        for i in range(len(input_shapes)):
            bw_ops.append(("MEM", ((B, H),))) # fill
    
    elif "mul" in name:
        opname = "VECmul"
        if len(input_shapes) == 1:
            opname += "u"
        B, H = read_vec_shape(input_shapes, output_shape)
        fw_ops.append((opname, (B, H)))
        for i in range(len(input_shapes)):
            bw_ops.append((opname, (B, H)))

    elif "div" in name:
        opname = "VECdiv"
        if len(input_shapes) == 1:
            opname += "u"
        B, H = read_vec_shape(input_shapes, output_shape)
        fw_ops.append((opname, (B, H)))
        for i in range(len(input_shapes)):
            bw_ops.append((opname, (B, H)))

    elif "pow" in name:
        opname = "VECpow"
        if len(input_shapes) == 1:
            opname += "u"
        B, H = read_vec_shape(input_shapes, output_shape)
        fw_ops.append((opname, (B, H)))
        bw_ops.append((opname, (B, H)))
        bw_ops.append(("VECmul", (B, H)))
        bw_ops.append(("VECmulu", (B, H)))

    elif "tanh" in name:
        opname = "VECtanh"
        B, H = read_vec_shape(input_shapes, output_shape)
        fw_ops.append((opname, (B, H)))
        bw_ops.append(("VECmulu", (B, H)))
        bw_ops.append(("VECaddu", (B, H)))

    elif "softmax" in name:
        opname = "VECsoftmax"
        B, H = read_vec_shape(input_shapes, output_shape)
        fw_ops.append((opname, (B, H)))
        bw_ops.append(("VECmul", (B, H)))
        bw_ops.append(("VECsoftmax", (B, H)))

    elif "_ln_" in name or "layer_norm" in name or "layernorm" in name:
        opname = "VECln"
        B, H = read_vec_shape(input_shapes, output_shape)
        fw_ops.append((opname, (B, H)))
        bw_ops.append(("VECmul", (B, H)))
        acc_ops.append(("VECadd", [1, H]))
        acc_ops.append(("VECadd", [1, H]))

    elif "gelu" in name:
        opname = "VECgelu"
        B, H = read_vec_shape(input_shapes, output_shape)
        fw_ops.append((opname, (B, H)))
        bw_ops.append((opname, (B, H)))

    elif "relu" in name or "_act" in name:
        opname = "VECrelu"
        B, H = read_vec_shape(input_shapes, output_shape)
        fw_ops.append((opname, (B, H)))
        bw_ops.append((opname, (B, H)))

    elif "word_embeddings" in name or "transformer_wte" in name or "model_decoder_embed_tokens" in name or "relative_attention_bias" in name:
        opname = "EMBEDDING"
        B, H = read_vec_shape(input_shapes, output_shape)
        fw_ops.append(("MEM", (*input_shapes, output_shape)))
        bw_ops.append(("MEM", ((vocab_size, H),))) # fill
        bw_ops.append(("MEM", ((vocab_size, H),))) # grad
        acc_ops.append(("VECadd", [1, H*vocab_size]))

    elif "where" in name:
        opname = "where"
        fw_ops.append(("MEM", (*input_shapes, output_shape)))
        bw_ops.append(("MEM", (*input_shapes,))) # fill
        bw_ops.append(("MEM", (*input_shapes,))) # grad

    elif "drop" in name:
        if is_train:
            opname = "dropout"
            fw_ops.append(("MEM", (*input_shapes, output_shape)))
            bw_ops.append(("MEM", (*input_shapes,))) # fill
            bw_ops.append(("MEM", (*input_shapes,))) # grad
        else:
            opname = "misc"
    
    elif "contiguous" in name:
        opname = "contiguous"
        fw_ops.append(("MEM", (*input_shapes, output_shape)))
        bw_ops.append(("MEM", (*input_shapes,))) # fill
        bw_ops.append(("MEM", (*input_shapes,))) # grad

    # for moe
    elif "rsqrt" in name:
        B, H = read_vec_shape(input_shapes, output_shape)

        opname = "rsqrt"
        fw_ops.append(("MEM", (*input_shapes, output_shape)))
        bw_ops.append(("MEM", (*input_shapes, output_shape)))

    elif "mean" in name:
        opname = "mean"
        fw_ops.append(("MEM", (*input_shapes, output_shape)))
        bw_ops.append(("MEM", (*input_shapes,))) # fill
        bw_ops.append(("MEM", (*input_shapes,))) # grad

    elif "count_nonzero" in name:
        opname = "count"
        fw_ops.append(("MEM", (*input_shapes, output_shape)))
        bw_ops.append(("MEM", (*input_shapes,))) # fill
        bw_ops.append(("MEM", (*input_shapes,))) # grad

    elif "cumsum" in name:
        opname = "cumsum"
        fw_ops.append(("MEM", (*input_shapes, output_shape)))
        bw_ops.append(("MEM", (*input_shapes,))) # fill
        bw_ops.append(("MEM", (*input_shapes,))) # grad

    elif "getitem" in name:
        opname = "getitem"
        fw_ops.append(("MEM", (*input_shapes, output_shape)))
        bw_ops.append(("MEM", (*input_shapes,))) # fill
        bw_ops.append(("MEM", (*input_shapes,))) # grad

    elif "setitem" in name:
        opname = "setitem"
        fw_ops.append(("MEM", (*input_shapes, output_shape)))
        bw_ops.append(("MEM", (*input_shapes,))) # fill
        bw_ops.append(("MEM", (*input_shapes,))) # grad

    elif "pad" in name:
        opname = "pad"
        fw_ops.append(("MEM", (*input_shapes, output_shape)))
        bw_ops.append(("MEM", (*input_shapes,))) # fill
        bw_ops.append(("MEM", (*input_shapes,))) # grad

    elif "output" in name:
        opname = "output"

    elif "reduce_from_tensor_model_parallel_region" in name:
        opname = "allreduce"
        buf_size = multiplyList(output_shape)
        fw_ops.append(("ALLREDUCE", (buf_size,)))
        bw_ops.append(("ALLREDUCE", (buf_size,)))

    elif "all_reduce_for_fx_cross_entropy" in name:
        opname = "allreduce"
        buf_size = multiplyList(output_shape)
        fw_ops.append(("ALLREDUCE", (buf_size,)))

    else:
        opname = "misc"

    # skip misc 1,1 vec ops
    if opname.startswith("VEC") and (B, H) == (1,1):
        opname = "misc"
        fw_ops = []
        bw_ops = []
        acc_ops = []
    
    if not is_train:
        bw_ops = []
        acc_ops = []

    fw_ops = [list(x) for x in fw_ops]
    bw_ops = [list(x) for x in bw_ops]
    acc_ops = [list(x) for x in acc_ops]

    return pd.Series((opname, fw_ops, bw_ops, acc_ops))

def parse_dependency(dependents):
    dependents = dependents[1:-1]
    if len(dependents) == 0:
        return set()
    dependents = dependents.split(", ")
    dependents = [d[:d.find(':')] for d in dependents]
    return set(dependents)

def parse_trace(
            input_csv, 
            bench, 
            is_train, 
            fusion=False,
            distributed=False,
            dp_degree=1,
            pp_degree=1,
            pp_num_microbatch=1,
            tp_degree=1,
        ):
    df = pd.read_csv(input_csv, converters={"input_shapes": ast.literal_eval, 
                                            "output_shape": ast.literal_eval,}
                    )

    entry = df.iloc[-1] # output
    vocab_size = entry.input_shapes[0][-1] # check for gpt
    columns = list(df.columns)
    df[["OpName", "FwOps", "BwOps", "AccOps"]] = df.apply(lambda x: parse_ops(x.iloc[columns.index("Name")], x.iloc[columns.index("input_shapes")], x.iloc[columns.index("output_shape")], x.iloc[columns.index("meta")], vocab_size, x.iloc[columns.index("input_contiguous")], is_train), axis=1)

    df["Prev"] = df.apply(lambda x: parse_dependency(x.iloc[columns.index("_input_nodes")]), axis=1)
    df["Next"] = df.apply(lambda x: parse_dependency(x.iloc[columns.index("users")]), axis=1)

    df = df.rename(columns={
                    "input_shapes" : "InputShapes",
                    "output_shape" : "OutputShape"
                })
    
    columns = ["Name", "OpName", "FwOps", "BwOps", "AccOps", "Prev", "Next", "InputShapes", "OutputShape"]
    df = df[columns]

    if fusion:
        df = fuse_parse(df)

    if distributed:
        if dp_degree > 1:
            result_df = pd.DataFrame(columns=df.columns)
            for index, row in df.iterrows():
                if len(row["FwOps"]) > 0:
                    if row["FwOps"][0][0] == 'VECln':
                        B, H = row["FwOps"][0][1]
                        new_row = {
                                    'Name': row["Name"] + "_grad", 
                                    'OpName': 'allreduce',
                                    'FwOps': [],
                                    'BwOps': [['ALLREDUCE_ASYNC', (H * 2,)]],
                                    'AccOps': [],
                                    'InputShapes': [],
                                    'OutputShape': [],
                                    'Prev' : set(),
                                    'Next' : set()
                                    }
                        result_df = pd.concat([result_df, pd.DataFrame([new_row])], ignore_index=True)
                    elif row["FwOps"][0][0] == 'Linear':
                        B, I, O = row["FwOps"][0][1]
                        new_row = {
                                    'Name': row["Name"] + "_grad", 
                                    'OpName': 'allreduce',
                                    'FwOps': [],
                                    'BwOps': [['ALLREDUCE_ASYNC', (I * O + O,)]],
                                    'AccOps': [],
                                    'InputShapes': [],
                                    'OutputShape': [],
                                    'Prev' : set(),
                                    'Next' : set()
                                    }
                        result_df = pd.concat([result_df, pd.DataFrame([new_row])], ignore_index=True)
                    elif row['Name'] == 'transformer_wte':
                        B, W, H = row["OutputShape"]
                        new_row = {
                                    'Name': row["Name"] + "_grad", 
                                    'OpName': 'allreduce',
                                    'FwOps': [],
                                    'BwOps': [['ALLREDUCE_ASYNC', (W * H,)]],
                                    'AccOps': [],
                                    'InputShapes': [],
                                    'OutputShape': [],
                                    'Prev' : set(),
                                    'Next' : set()
                                    }
                        result_df = pd.concat([result_df, pd.DataFrame([new_row])], ignore_index=True)
                # Append the current row
                result_df = pd.concat([result_df, pd.DataFrame([row])], ignore_index=True)
            df = result_df

        elif pp_degree > 1:
            result_df = pd.DataFrame(columns=df.columns)
            for index, row in df.iterrows():
                result_df = pd.concat([result_df, pd.DataFrame([row])], ignore_index=True)
                if len(row["FwOps"]) > 0:
                    if row['Name'] == 'add_15':
                        B, S, H = row["OutputShape"]
                        new_row = {
                                    'Name': "sendrecv", 
                                    'OpName': 'sendrecv',
                                    'FwOps': [['SENDRECV', (B * S * H,)]],
                                    'BwOps': [['SENDRECV', (B * S * H,)]],
                                    'AccOps': [],
                                    'InputShapes': [],
                                    'OutputShape': [],
                                    'Prev' : set(),
                                    'Next' : set()
                                    }
                        result_df = pd.concat([result_df, pd.DataFrame([new_row])], ignore_index=True)
            df = result_df

        elif tp_degree > 1:
            # Initialize an empty DataFrame to store the result
            result_df = pd.DataFrame(columns=df.columns)

            # Iterate over the original DataFrame
            for index, row in df.iterrows():

                if len(row["FwOps"]) == 0:
                    result_df = pd.concat([result_df, pd.DataFrame([row])], ignore_index=True)
                else:
                    if row['Name'] == 'transformer_wte':
                        B, S, H = row["OutputShape"]
                        W, H = row["BwOps"][0][1][0]
                
                        # split the tensors
                        fw_ops = []
                        bw_ops = []
                        acc_ops = []

                        W = W // tp_degree
                        
                        bw_ops.append(("MEM", ((W, H),))) # fill
                        bw_ops.append(("MEM", ((W, H),))) # grad
                        acc_ops.append(("VECadd", [1, H*W]))

                        row['BwOps'] = bw_ops
                        row['AccOps'] = acc_ops

                        result_df = pd.concat([result_df, pd.DataFrame([row])], ignore_index=True)
                        
                        new_row = {
                                    'Name': "reduce_from_tensor_model_parallel_region", 
                                    'OpName': 'allreduce',
                                    'FwOps': [['ALLREDUCE', (B * S * H,)]],
                                    'BwOps': [['ALLREDUCE', (B * S * H,)]],
                                    'AccOps': [],
                                    'InputShapes': [],
                                    'OutputShape': [],
                                    'Prev' : set(),
                                    'Next' : set()
                                    }
                        result_df = pd.concat([result_df, pd.DataFrame([new_row])], ignore_index=True)

                    elif row['Name'] == 'addmm':
                        B, I, O = row["FwOps"][0][1]

                        fw_ops = []
                        bw_ops = []
                        acc_ops = []

                        O = O // tp_degree

                        fw_ops.append(("Linear", (B, I, O)))
                        bw_ops.append(("Linear", (B, O, I)))
                        bw_ops.append(("Linear", (O, B, I)))
                        acc_ops.append(("VECadd", [1, O*I]))
                        acc_ops.append(("VECadd", [1, O]))

                        row['FwOps'] = fw_ops
                        row['BwOps'] = bw_ops
                        row['AccOps'] = acc_ops

                        row["InputShapes"] = [(O,), (B, I), (B, O)]
                        row["OutputShape"] = (B, O)

                        result_df = pd.concat([result_df, pd.DataFrame([row])], ignore_index=True)

                    elif row['Name'] == 'addmm_1':
                        B, I, O = row["FwOps"][0][1]

                        fw_ops = []
                        bw_ops = []
                        acc_ops = []

                        I = I // tp_degree

                        fw_ops.append(("Linear", (B, I, O)))
                        bw_ops.append(("Linear", (B, O, I)))
                        bw_ops.append(("Linear", (O, B, I)))
                        acc_ops.append(("VECadd", [1, O*I]))
                        acc_ops.append(("VECadd", [1, O]))

                        row['FwOps'] = fw_ops
                        row['BwOps'] = bw_ops
                        row['AccOps'] = acc_ops

                        row["InputShapes"] = [(O,), (B, I), (B, O)]
                        row["OutputShape"] = (B, O)

                        result_df = pd.concat([result_df, pd.DataFrame([row])], ignore_index=True)

                        new_row = {
                                    'Name': "reduce_from_tensor_model_parallel_region_1", 
                                    'OpName': 'allreduce',
                                    'FwOps': [['ALLREDUCE', (B * O,)]],
                                    'BwOps': [['ALLREDUCE', (B * O,)]],
                                    'AccOps': [],
                                    'InputShapes': [],
                                    'OutputShape': [],
                                    'Prev' : set(),
                                    'Next' : set()
                                    }
                        result_df = pd.concat([result_df, pd.DataFrame([new_row])], ignore_index=True)

                    elif row['Name'] == 'addmm_2':
                        B, I, O = row["FwOps"][0][1]

                        fw_ops = []
                        bw_ops = []
                        acc_ops = []

                        O = O // tp_degree

                        fw_ops.append(("Linear", (B, I, O)))
                        bw_ops.append(("Linear", (B, O, I)))
                        bw_ops.append(("Linear", (O, B, I)))
                        acc_ops.append(("VECadd", [1, O*I]))
                        acc_ops.append(("VECadd", [1, O]))

                        row['FwOps'] = fw_ops
                        row['BwOps'] = bw_ops
                        row['AccOps'] = acc_ops

                        row["InputShapes"] = [(O,), (B, I), (B, O)]
                        row["OutputShape"] = (B, O)

                        result_df = pd.concat([result_df, pd.DataFrame([row])], ignore_index=True)

                    elif row['Name'] == 'addmm_3':
                        B, I, O = row["FwOps"][0][1]

                        fw_ops = []
                        bw_ops = []
                        acc_ops = []

                        I = I // tp_degree

                        fw_ops.append(("Linear", (B, I, O)))
                        bw_ops.append(("Linear", (B, O, I)))
                        bw_ops.append(("Linear", (O, B, I)))
                        acc_ops.append(("VECadd", [1, O*I]))
                        acc_ops.append(("VECadd", [1, O]))

                        row['FwOps'] = fw_ops
                        row['BwOps'] = bw_ops
                        row['AccOps'] = acc_ops

                        row["InputShapes"] = [(O,), (B, I), (B, O)]
                        row["OutputShape"] = (B, O)

                        result_df = pd.concat([result_df, pd.DataFrame([row])], ignore_index=True)

                        new_row = {
                                    'Name': "reduce_from_tensor_model_parallel_region_2", 
                                    'OpName': 'allreduce',
                                    'FwOps': [['ALLREDUCE', (B * O,)]],
                                    'BwOps': [['ALLREDUCE', (B * O,)]],
                                    'AccOps': [],
                                    'InputShapes': [],
                                    'OutputShape': [],
                                    'Prev' : set(),
                                    'Next' : set()
                                    }
                        result_df = pd.concat([result_df, pd.DataFrame([new_row])], ignore_index=True)

                    elif row['Name'] == 'lm_head':
                        B, I, O = row["FwOps"][0][1]

                        fw_ops = []
                        bw_ops = []
                        acc_ops = []

                        O = O // tp_degree

                        fw_ops.append(("Linear", (B, I, O)))
                        bw_ops.append(("Linear", (B, O, I)))
                        bw_ops.append(("Linear", (O, B, I)))
                        acc_ops.append(("VECadd", [1, O*I]))
                        acc_ops.append(("VECadd", [1, O]))

                        row['FwOps'] = fw_ops
                        row['BwOps'] = bw_ops
                        row['AccOps'] = acc_ops

                        row["InputShapes"] = [(O,), (B, I), (B, O)]
                        row["OutputShape"] = (B, O)

                        result_df = pd.concat([result_df, pd.DataFrame([row])], ignore_index=True)

                    elif row['Name'] == 'matmul':
                        B, H, S, D = row["InputShapes"][0]

                        H = H // tp_degree

                        fw_ops = []
                        bw_ops = []

                        fw_ops.append(("BMM", (B*H, S, D, S)))
                        bw_ops.append(("BMM", (B*H, S, S, D)))
                        bw_ops.append(("BMM", (B*H, D, S, S)))

                        row['FwOps'] = fw_ops
                        row['BwOps'] = bw_ops

                        row["InputShapes"] = [(B,H,S,D), (B,H,D,S)]
                        row["OutputShape"] = (B,H,S,S)

                        result_df = pd.concat([result_df, pd.DataFrame([row])], ignore_index=True)

                    elif row['Name'] == 'matmul_1':
                        B, H, S, D = row["OutputShape"]

                        H = H // tp_degree

                        fw_ops = []
                        bw_ops = []

                        fw_ops.append(("BMM", (B*H, S, D, S)))
                        bw_ops.append(("BMM", (B*H, S, S, D)))
                        bw_ops.append(("BMM", (B*H, D, S, S)))

                        row['FwOps'] = fw_ops
                        row['BwOps'] = bw_ops

                        row["InputShapes"] = [(B,H,S,D), (B,H,D,S)]
                        row["OutputShape"] = (B,H,S,S)

                        result_df = pd.concat([result_df, pd.DataFrame([row])], ignore_index=True)

                    elif row['Name'] == 'truediv':
                        B, H, S, S = row["OutputShape"]

                        H = H // tp_degree

                        fw_ops = []
                        bw_ops = []

                        fw_ops.append(("VECdiv", (B*H*S, S)))
                        bw_ops.append(("VECdiv", (B*H*S, S)))
                        bw_ops.append(("VECdiv", (B*H*S, S)))

                        row['FwOps'] = fw_ops
                        row['BwOps'] = bw_ops

                        row["InputShapes"] = [(B,H,S,S), (S,)]
                        row["OutputShape"] = (B, H, S, S)

                        result_df = pd.concat([result_df, pd.DataFrame([row])], ignore_index=True)

                    elif row['Name'] == 'transformer_h_0_attn_attn_dropout':
                        B, H, S, S = row["OutputShape"]

                        H = H // tp_degree

                        fw_ops = []
                        bw_ops = []

                        fw_ops.append(("MEM", ((B,H,S,S), (B,H,S,S))))
                        bw_ops.append(("MEM", ((B,H,S,S),)))
                        bw_ops.append(("MEM", ((B,H,S,S),)))

                        row['FwOps'] = fw_ops
                        row['BwOps'] = bw_ops

                        row["InputShapes"] = [(B,H,S,S),]
                        row["OutputShape"] = (B, H, S, S)

                        result_df = pd.concat([result_df, pd.DataFrame([row])], ignore_index=True)

                    elif row['Name'] == 'contiguous':
                        B, S, H, D = row["OutputShape"]

                        H = H // tp_degree

                        fw_ops = []
                        bw_ops = []

                        # fw [['MEM', ((4, 1024, 20, 64), (4, 1024, 20, 64))]]	
                        # bw [['MEM', ((4, 1024, 20, 64),)], ['MEM', ((4, 1024, 20, 64),)]]
                        fw_ops.append(("MEM", ((B, S, H, D), (B, S, H, D))))
                        bw_ops.append(("MEM", ((B, S, H, D),)))
                        bw_ops.append(("MEM", ((B, S, H, D),)))

                        row['FwOps'] = fw_ops
                        row['BwOps'] = bw_ops

                        row["InputShapes"] = [(B,S,H,D),]
                        row["OutputShape"] = (B,S,H,D)

                        result_df = pd.concat([result_df, pd.DataFrame([row])], ignore_index=True)

                    elif row['Name'] == 'where':
                        B, H, S, S = row["OutputShape"]

                        H = H // tp_degree

                        fw_ops = []
                        bw_ops = []

                        fw_ops.append(("MEM", ((1, 1, S, S), (B, H, S, S), (1,), (B, H, S, S))))
                        
                        # bw ops [['MEM', ((1, 1, 1024, 1024), (4, 20, 1024, 1024), (1,))], ['MEM', ((1, 1, 1024, 1024), (4, 20, 1024, 1024), (1,))]]
                        bw_ops.append(("MEM", ((1, 1, S, S), (B, H, S, S), (1,),)))
                        bw_ops.append(("MEM", ((1, 1, S, S), (B, H, S, S), (1,),)))

                        row['FwOps'] = fw_ops
                        row['BwOps'] = bw_ops

                        row["InputShapes"] = [(1,1,S,S),(B,H,S,S),(1,)]
                        row["OutputShape"] = (B,H,S,S)

                        result_df = pd.concat([result_df, pd.DataFrame([row])], ignore_index=True)

                    elif row['Name'] == 'softmax':
                        B, H, S, S = row["OutputShape"]

                        H = H // tp_degree

                        fw_ops = []
                        bw_ops = []

                        fw_ops.append(("VECsoftmax", (B*H*S, S)))
                        bw_ops.append(("VECmul", (B*H*S, H)))
                        bw_ops.append(("VECsoftmax", (B*H*S, H)))

                        row['FwOps'] = fw_ops
                        row['BwOps'] = bw_ops

                        row["InputShapes"] = [(B,H,S,S),]
                        row["OutputShape"] = (B,H,S,S)

                        result_df = pd.concat([result_df, pd.DataFrame([row])], ignore_index=True)

                    elif row["Name"] == "getitem_5" or row["Name"] == "getitem_6" or row["Name"] == "getitem_7":
                        print(f"\noutputShape: {row['OutputShape']} for name: {row['Name']}\n")
                        B, S, H = row["OutputShape"]
                        H = H // tp_degree
                        fw_ops = []
                        fw_ops.append(("MEM", ((1,), (B, S, H))))
                        row['FwOps'] = fw_ops
                        row["InputShapes"] = [(O,), (B, I), (B, O)]
                        row["OutputShape"] = (B, S, H)

                        result_df = pd.concat([result_df, pd.DataFrame([row])], ignore_index=True)

                    elif row['Name'] in ['mul', 'mul_1', 'mul_2', 'mul_3']:
                        print(f"\noutputShape: {row['OutputShape']} for name: {row['Name']}\n")
                        B, S, H = row["OutputShape"]
                        H = H // tp_degree
                        fw_ops = [('VECmulu', (B * S, H))]
                        bw_ops = [('VECmulu', (B * S, H))]
                        row['FwOps'] = fw_ops
                        row['BwOps'] = bw_ops
                        result_df = pd.concat([result_df, pd.DataFrame([row])], ignore_index=True)

                    elif row['Name'] == 'pow_2':
                        B, S, H = row["OutputShape"]
                        H = H // tp_degree
                        fw_ops = [('VECpowu', (B * S, H))]
                        bw_ops = [
                            ('VECpowu', (B * S, H)),
                            ('VECmul', (B * S, H)),
                            ('VECmulu', (B * S, H))
                        ]
                        row['FwOps'] = fw_ops
                        row['BwOps'] = bw_ops
                        result_df = pd.concat([result_df, pd.DataFrame([row])], ignore_index=True)

                    elif row['Name'] == 'add_12':
                        B, S, H = row["OutputShape"]
                        H = H // tp_degree
                        fw_ops = [('VECadd', (B * S, H))]
                        bw_ops = [
                            ('MEM', ((B * S, H),)),
                            ('MEM', ((B * S, H),))
                        ]
                        row['FwOps'] = fw_ops
                        row['BwOps'] = bw_ops
                        result_df = pd.concat([result_df, pd.DataFrame([row])], ignore_index=True)

                    elif row['Name'] == 'tanh':
                        B, S, H = row["OutputShape"]
                        H = H // tp_degree
                        fw_ops = [('VECtanh', (B * S, H))]
                        bw_ops = [
                            ('VECmulu', (B * S, H)),
                            ('VECaddu', (B * S, H))
                        ]
                        row['FwOps'] = fw_ops
                        row['BwOps'] = bw_ops
                        result_df = pd.concat([result_df, pd.DataFrame([row])], ignore_index=True)

                    elif row['Name'] == 'add_13':
                        B, S, H = row["OutputShape"]
                        H = H // tp_degree
                        fw_ops = [('VECaddu', (B * S, H))]
                        bw_ops = [('MEM', ((B * S, H),))]
                        row['FwOps'] = fw_ops
                        row['BwOps'] = bw_ops
                        result_df = pd.concat([result_df, pd.DataFrame([row])], ignore_index=True)

                    else:
                        result_df = pd.concat([result_df, pd.DataFrame([row])], ignore_index=True)
            df = result_df

        else:
            assert(0)

    return df

def generate_tp(model_name, tp_degree, global_batch_size, result_dir):
    # hyperparameters
    tmp_width = tp_degree
    micro_batch_size = global_batch_size
    if "gpt2" in model_name:
        seq_len = 1024
        n_layer = 36
    elif "gpt3" in model_name:
        seq_len = 2048
        n_layer = 24
    else:
        assert(0)
    source_file = result_dir/f"parse/{model_name}_tmp_{tmp_width}-train-{seq_len}-{micro_batch_size}.csv"
    out_file = result_dir/f"parse/{model_name}_1_{tp_degree}_1-train-{seq_len}-{global_batch_size}.csv"
    df = pd.read_csv(source_file)
    df.to_csv(out_file, index=False)
    return df

def generate_dp(model_name, dp_degree, global_batch_size, result_dir):
    # hyperparameters
    micro_batch_size = global_batch_size // dp_degree

    if "gpt2" in model_name:
        seq_len = 1024
        n_layer = 36
    elif "gpt3" in model_name:
        seq_len = 2048
        n_layer = 24
    else:
        assert(0)

    source_file = result_dir/f"parse/{model_name}_tmp_1-train-{seq_len}-{micro_batch_size}.csv"
    out_file = result_dir/f"parse/{model_name}_{dp_degree}_1_1-train-{seq_len}-{global_batch_size}.csv"

    df = pd.read_csv(source_file, converters={"FwOps": ast.literal_eval, "BwOps": ast.literal_eval, "AccOps": ast.literal_eval, "InputShapes": ast.literal_eval, "OutputShape": ast.literal_eval})

    # drop tensor parallel
    df = df[~df["Name"].str.contains("tensor_model_parallel")]

    new_rows = []

    for index, row in df.iterrows():
        row = dict(row)
        new_rows.append(row)
        if row["AccOps"] != []:
            # accumulate gradients to communicate
            buf_size = 0
            for opname, args in row["AccOps"]:
                buf_size += args[1]
            grad_sync_row = {
                "Name" : row["Name"]+"_grad",
                "OpName" : "allreduce",
                "FwOps" : [],
                "BwOps" : [['ALLREDUCE_ASYNC', (buf_size,)]],
                "AccOps" : [],
                "Prev" : [],
                "Next" : [],
                "InputShapes" : [],
                "OutputShape" : [],
            }
            new_rows.append(grad_sync_row)

    df = pd.DataFrame(new_rows)
    df.to_csv(out_file, index=False)

    return df

def generate_pp(model_name, pp_degree, global_batch_size, result_dir, num_micro_batch):
    # hyperparameters
    micro_batch_size = global_batch_size // num_micro_batch

    if "gpt2" in model_name:
        seq_len = 1024
    elif "gpt3" in model_name:
        seq_len = 2048
    else:
        assert(0)

    source_file = result_dir/f"parse/{model_name}_tmp_1-train-{seq_len}-{micro_batch_size}.csv"
    out_file = result_dir/f"parse/{model_name}_1_1_{pp_degree}-train-{seq_len}-{global_batch_size}.csv"

    df = pd.read_csv(source_file, converters={"FwOps": ast.literal_eval, "BwOps": ast.literal_eval, "AccOps": ast.literal_eval, "InputShapes": ast.literal_eval, "OutputShape": ast.literal_eval})

    # drop tensor parallel
    df = df[~df["Name"].str.contains("tensor_model_parallel")]

    # drop all reduce
    df = df[~df["Name"].str.contains("all_reduce")]

    new_rows = []

    for index, row in df.iterrows():
        row = dict(row)
        new_rows.append(row)
        if row["Name"] == "make_viewless_tensor_1":
            # accumulate gradients to communicate
            buf_size = multiplyList(row["OutputShape"])
            sendrecv_row = {
                "Name" : "sendrecv",
                "OpName" : "sendrecv",
                "FwOps" : [['SENDRECV', (buf_size,)]],
                "BwOps" : [['SENDRECV', (buf_size,)]],
                "AccOps" : [],
                "Prev" : [],
                "Next" : [],
                "InputShapes" : [],
                "OutputShape" : [],
            }
            new_rows.append(sendrecv_row)

    df = pd.DataFrame(new_rows)
    df.to_csv(out_file, index=False)

    return df