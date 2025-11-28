#!/usr/bin/env python
# ## Boundless DAS

# Adapted from Zhengxuan Wu's tutorial in `pyvene`.

# ### Overview
# 
# This script produces the figures tracking performance as a function of the CL loss on the Boundless DAS example.


import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import time

import pyvene

import torch
from tqdm import tqdm, trange
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from torch.nn import CrossEntropyLoss
from tutorial_price_tagging_utils import (
    factual_sampler,
    bound_alignment_sampler,
    lower_bound_alignment_example_sampler,
)

from pyvene import (
    IntervenableModel,
    BoundlessRotatedSpaceIntervention,
    RepresentationConfig,
    IntervenableConfig,
)
from pyvene import create_llama
from pyvene import set_seed, count_parameters

from divergence_utils import (
    get_cor_mtx, get_mse_mtx, optimal_pairs, sample_without_replacement,
    visualize_states, collect_divergences,
)


seed = 42
set_seed(seed)

cl_eps_range = [
    0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 16.0, 32.0, 64.0
]
do_rotate = False # Determine whether to rotate the intervention vectors
    # and applying the boundary mask before calculating the loss with the CL vectors.
    # If False, the intervention vectors are not rotated and the boundary mask is
    # not applied before calculating the loss with the CL vectors.
intrv_type = "boundless" # Determine whether to use the Boundless DAS intervention or the Hacky DAS intervention.
mask_dims = 3 # Determine the dimensions of the intervention mask. Only applies
    # if using the Hacky DAS intervention.
n_seeds = 3 # will repeat the whole experiment this many times
train_epochs = 3
train_batch_size = 32
cl_collection_batch_size = 128
eval_batch_size = 64
train_gradient_accumulation_steps = 4
n_divergence_samples = 5
use_numpy = False
debug = False
save_actvs_dir = "./cl_boundless_actvs/"
save_stamp = datetime.fromtimestamp(time.time()).strftime("%Y%m%d_%H%M%S")
save_stamp = str(save_stamp)

for arg in sys.argv[1:]:
    if "debug" in arg:
        debug = "true" in arg.lower()
print(f"Debug mode: {debug}")

config, tokenizer, llama = create_llama()
_ = llama.to("cuda")  # single gpu
_ = llama.eval()  # always no grad on the model

if save_actvs_dir is not None and not debug:
    if not os.path.exists(save_actvs_dir):
        os.makedirs(save_actvs_dir)

# ### Create training dataset for our trainable intervention (Boundless DAS)

###################
# data loaders
###################
tot_samples = 10000
n_train = 8000
n_eval = 1000
n_test = 1000
if debug:
    n_train = 100
    n_eval = 100
    n_test = 100

raw_data = bound_alignment_sampler(
    tokenizer,
    tot_samples,
    [lower_bound_alignment_example_sampler]
)

raw_train = (
    raw_data[0][:n_train],
    raw_data[1][:n_train],
    raw_data[2][:n_train],
    raw_data[3][:n_train],
    raw_data[4][:n_train],
)
raw_eval = (
    raw_data[0][n_train:n_train+n_eval],
    raw_data[1][n_train:n_train+n_eval],
    raw_data[2][n_train:n_train+n_eval],
    raw_data[3][n_train:n_train+n_eval],
    raw_data[4][n_train:n_train+n_eval],
)
raw_test = (
    raw_data[0][-n_test:],
    raw_data[1][-n_test:],
    raw_data[2][-n_test:],
    raw_data[3][-n_test:],
    raw_data[4][-n_test:],
)
train_dataset = Dataset.from_dict(
    {
        "input_ids": raw_train[0],
        "source_input_ids": raw_train[1],
        "labels": raw_train[2],
        "subspace_ids": raw_train[3],
        "cl_input_ids": raw_train[4],
        "indices": [i for i in range(len(raw_train[0]))],
    }
).with_format("torch")
train_dataloader = DataLoader(
    train_dataset,
    batch_size=cl_collection_batch_size,
    shuffle=False,
)
eval_dataset = Dataset.from_dict(
    {
        "input_ids": raw_eval[0],
        "source_input_ids": raw_eval[1],
        "labels": raw_eval[2],
        "subspace_ids": raw_eval[3],
        "cl_input_ids": raw_eval[4],
        "indices": [i for i in range(len(raw_eval[0]))],
    }
).with_format("torch")
eval_dataloader = DataLoader(
    eval_dataset,
    batch_size=cl_collection_batch_size,
    shuffle=False,
)
test_dataset = Dataset.from_dict(
    {
        "input_ids": raw_test[0],
        "source_input_ids": raw_test[1],
        "labels": raw_test[2],
        "subspace_ids": raw_test[3],
        "cl_input_ids": raw_test[4],
        "indices": [i for i in range(len(raw_test[0]))],
    }
).with_format("torch")
test_dataloader = DataLoader(
    test_dataset,
    batch_size=cl_collection_batch_size,
    shuffle=False,
)


# ### Boundless DAS on Position-aligned Tokens

def simple_boundless_das_position_config(
        model_type, intervention_type, layer, intrv_type="boundless",
):
    if intrv_type == "boundless":
        class_type = BoundlessRotatedSpaceIntervention
    elif intrv_type == "hacky":
        raise NotImplementedError("Hacky DAS is not implemented yet.")
        #class_type = HackyRotatedSpaceIntervention
    else:
        raise ValueError(f"Invalid intervention type: {intrv_type}")
    config = IntervenableConfig(
        model_type=model_type,
        representations=[
            RepresentationConfig(
                layer,              # layer
                intervention_type,  # intervention type
            ),
        ],
        intervention_types=class_type,
    )
    return config


layer_num = 15
config = simple_boundless_das_position_config(
    type(llama), "block_output", layer_num, intrv_type=intrv_type
)
intervenable = IntervenableModel(config, llama)
intervenable.set_device("cuda")
intervenable.disable_model_gradients()

# Will use the comms_dict to collect the source and intervened vectors from
# the Boundless DAS intervention module.
# After each intervention, these vectors will be stored in the keys "source_vectors"
# and "intrv_vectors" dict keys respectively.
key = list(intervenable.interventions.keys())[0] # Get name of the intervention
das_object = intervenable.interventions[key][0] # Get the intervention object
if intrv_type == "hacky":
    das_object.set_masks(n_masks=3, mask_dims=mask_dims)
comms_dict = das_object.comms_dict # Get the comms dictionary


def collect_source_vectors(intervenable, dataloader, source_key="cl_input_ids"):
    source_vectors = []
    tqdm_iterator = tqdm(dataloader, desc="Collecting CL Vectors")
    with torch.no_grad():
        for inputs in tqdm_iterator:
            for k, v in inputs.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inputs[k] = v.to("cuda")
            _, _ = intervenable(
                {"input_ids": inputs["input_ids"]},
                [{"input_ids": inputs[source_key]}],
                {"sources->base": 80},  # swap 80th token
            )
            source_vectors.append(comms_dict["source_vectors"])
    return torch.vstack(source_vectors)

# We can collect the CL vectors by running the CL data as the source data and
# collecting the source vectors from the Boundless DAS intervention module.
print("Collecting train set cl vectors")
train_cl_vectors = collect_source_vectors(intervenable, train_dataloader).cpu()
print("Collecting test set cl vectors")
test_cl_vectors = collect_source_vectors(intervenable, test_dataloader).cpu()

test_dataloader = DataLoader(
    test_dataset,
    batch_size=eval_batch_size,
    shuffle=False,
)

combination_type = "full_match" 
cl_keys = ["full_match"]
train_cl_vector_dict = {cl_keys[0]: train_cl_vectors}
test_cl_vector_dict = {cl_keys[0]: test_cl_vectors}

train_cl_failure_dict = {cl_keys[0]: torch.zeros(n_train).bool()}
test_cl_failure_dict = {cl_keys[0]: torch.zeros(n_test).bool()}


# You can define your custom compute_metrics function.
def compute_metrics(eval_preds, eval_labels):
    total_count = 0
    correct_count = 0
    for eval_pred, eval_label in zip(eval_preds, eval_labels):
        actual_test_labels = eval_label[:, -1]
        pred_test_labels = torch.argmax(eval_pred[:, -1], dim=-1)
        correct_labels = actual_test_labels == pred_test_labels
        total_count += len(correct_labels)
        correct_count += correct_labels.sum().tolist()
    accuracy = round(correct_count / total_count, 2)
    return {"accuracy": accuracy}

def calculate_loss(logits, labels):
    shift_logits = logits[..., :, :].contiguous()
    shift_labels = labels[..., :].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss()
    shift_logits = shift_logits.view(-1, intervenable.model_config.vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)

    for k, v in intervenable.interventions.items():
        boundary_loss = 1.0 * v[0].intervention_boundaries.sum()
    loss += boundary_loss

    return loss

def cl_loss_fn(intrv_vectors, cl_vectors, loss_type="both"):
    """
    Calculate the loss between the intervention vectors and the cl vectors.
    """
    mse,cos = 0.0,0.0
    if loss_type in {"both","mse"}:
        mse = (intrv_vectors-cl_vectors).pow(2).mean()
    if loss_type in {"both","cos"}:
        cos = 1-torch.cosine_similarity(intrv_vectors,cl_vectors,dim=-1).mean()
    return mse+cos

def rotated_cl_loss(
        das_object,
        intrv_vectors,
        cl_vectors,
        subspaces=None,
        loss_type="both"
):
    """
    Calculate the loss between the intervention vectors and the cl vectors.
    """
    mask = das_object.get_boundary_mask(
        batch_size=intrv_vectors.shape[0],
        subspaces=subspaces,
        device=intrv_vectors.device,
        dtype=intrv_vectors.dtype
    )
    intrv_vectors = das_object.rotate_layer(intrv_vectors)*mask
    with torch.no_grad():
        c_vectors = das_object.rotate_layer(cl_vectors)*mask
    return cl_loss_fn(intrv_vectors, c_vectors.data, loss_type)

def calculate_cl_loss(
        intrv_vectors,
        inputs,
        cl_vector_dict,
        cl_failure_dict,
        das_object,
        do_rotate=True
):
    device = next(das_object.parameters()).data.device
    intrv_vectors = intrv_vectors.to(device)
    cl_loss = 0.0
    cl_idxs = inputs["indices"].long().to(device)
    subspaces = inputs["subspace_ids"].to(device)
    for cl_key in cl_vector_dict.keys():
        cl_vectors = cl_vector_dict[cl_key].to(device)[cl_idxs]
        cl_keeps = ~cl_failure_dict[cl_key].to(device)[cl_idxs]
        if do_rotate:
            cl_l = rotated_cl_loss(
                das_object,
                intrv_vectors[cl_keeps],
                cl_vectors[cl_keeps],
                subspaces=subspaces[cl_keeps],
                loss_type="both"
            )
        else:
            cl_l = cl_loss_fn(
                intrv_vectors[cl_keeps],
                cl_vectors[cl_keeps],
                loss_type="both"
            )
        cl_loss += cl_l
        cl_vector_dict[cl_key] = cl_vector_dict[cl_key].cpu()
        cl_failure_dict[cl_key] = cl_failure_dict[cl_key].cpu()
    return cl_loss / len(cl_vector_dict.keys())

metrics_dict = {
    "seed": [],
    "method": [],
    "cl_eps": [],
    "train_accuracy": [],
    "test_accuracy": [],
    "train_cl_loss": [],
    "train_actn_loss": [],
    "test_cl_loss": [],
    "test_actn_loss": [],
}

for _ in range(n_seeds):
    seed = seed + 12345
    torch.cuda.empty_cache()

    for cl_eps in cl_eps_range:
        print(f"Running CL epsilon: {cl_eps} - Seed: {seed}")
        set_seed(seed)
        epochs = train_epochs if not debug else 1
        layer_num = 15
        
        # Rebuild a new intervention module with the same configuration for each CL epsilon
        intervenable.cpu()
        das_object.cpu()
        del intervenable
        del das_object
        config = simple_boundless_das_position_config(
            type(llama), "block_output", layer_num
        )
        intervenable = IntervenableModel(config, llama)
        intervenable.set_device("cuda")
        intervenable.disable_model_gradients()
        
        key = list(intervenable.interventions.keys())[0] # Get name of the intervention
        das_object = intervenable.interventions[key][0] # Get the intervention object
        das_object.comms_dict = comms_dict # Reuse the same comms dictionary
    
        # Rebuild the training objects for each CL epsilon
        train_dataloader = DataLoader(
            train_dataloader.dataset,
            batch_size=train_batch_size,
            shuffle=True,
        )
        t_total = int(len(train_dataloader) * train_epochs)
        warm_up_steps = 0.1 * t_total
        optimizer_params = []
        for k, v in intervenable.interventions.items():
            optimizer_params += [{"params": v[0].rotate_layer.parameters()}]
            optimizer_params += [{"params": v[0].intervention_boundaries, "lr": 1e-2}]
        optimizer = torch.optim.Adam(optimizer_params, lr=1e-3)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warm_up_steps, num_training_steps=t_total
        )
        
        gradient_accumulation_steps = train_gradient_accumulation_steps
        target_total_step = len(train_dataloader) * epochs
        temperature_start = 50.0
        temperature_end = 0.1
        temperature_schedule = (
            torch.linspace(temperature_start, temperature_end, target_total_step)
            .to(torch.bfloat16)
            .to("cuda")
        )
        intervenable.set_temperature(temperature_schedule[0])
        
        
        # Train the model
        intervenable.model.train()  # train enables drop-off but no grads
        print("llama trainable parameters: ", count_parameters(intervenable.model))
        print("intervention trainable parameters: ", intervenable.count_parameters())
        train_iterator = trange(0, int(epochs), desc="Epoch")
        total_step = 0
        for epoch in train_iterator:
            epoch_iterator = tqdm(
                train_dataloader, desc=f"Epoch: {epoch}", position=0, leave=True
            )
            train_intrv_hstates = []
            train_cl_hstates = []
            train_actn_loss = 0.0
            train_cl_loss = 0.0
            train_accuracy = 0.0
            n_train_steps = len(epoch_iterator)
            for step, inputs in enumerate(epoch_iterator):
                for k, v in inputs.items():
                    if v is not None and isinstance(v, torch.Tensor):
                        inputs[k] = v.to("cuda")
                comms_dict["subspaces"] = inputs["subspace_ids"]
                b_s = inputs["input_ids"].shape[0]
                _, counterfactual_outputs = intervenable(
                    {"input_ids": inputs["input_ids"]},
                    [{"input_ids": inputs["source_input_ids"]}],
                    {"sources->base": 80},  # swap 80th token
                )
                eval_metrics = compute_metrics(
                    [counterfactual_outputs.logits], [inputs["labels"]]
                )
        
                # loss and backprop
                loss = calculate_loss(counterfactual_outputs.logits, inputs["labels"])
                actn_loss = loss
        
                # CL Loss
                cl_loss = calculate_cl_loss(
                    intrv_vectors=comms_dict["intrv_vectors"],
                    inputs=inputs,
                    cl_vector_dict=train_cl_vector_dict,
                    cl_failure_dict=train_cl_failure_dict,
                    das_object=das_object,
                    do_rotate=do_rotate,
                )
                loss = loss + cl_eps*cl_loss

                train_actn_loss += actn_loss.item()/n_train_steps
                train_cl_loss += cl_loss.item()/n_train_steps
                train_accuracy += eval_metrics["accuracy"]/n_train_steps

                # Print Loss
                loss_str = round(actn_loss.item(), 2)
                cl_loss_str = round(cl_loss.item(), 4)
                epoch_iterator.set_postfix({
                    "loss": loss_str,
                    "acc": eval_metrics["accuracy"],
                    "CL": cl_loss_str
                })
        
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps
                loss.backward()
                if total_step % gradient_accumulation_steps == 0:
                    if not (gradient_accumulation_steps > 1 and total_step == 0):
                        optimizer.step()
                        scheduler.step()
                        intervenable.set_zero_grad()
                        intervenable.set_temperature(
                            temperature_schedule[min(total_step,len(temperature_schedule))]
                        )
                total_step += 1
                
                train_intrv_hstates.append(comms_dict["intrv_vectors"].cpu())
                train_cl_hstates.append(train_cl_vector_dict[cl_keys[0]][inputs["indices"].cpu().long()].cpu())
            print("Train Acc:", train_accuracy)
            print("\tActn Loss:", train_actn_loss)
            print("\tCL Loss:", train_cl_loss)
            torch.cuda.empty_cache()
        train_eval_metrics = eval_metrics
        torch.cuda.empty_cache()

        metrics_dict["seed"].append(seed)
        metrics_dict["cl_eps"].append(cl_eps)
        metrics_dict["train_accuracy"].append(train_accuracy)
        metrics_dict["train_actn_loss"].append(train_actn_loss)
        metrics_dict["train_cl_loss"].append(train_cl_loss)

        print("Train Acc:", train_accuracy)
        print("\tActn Loss:", train_actn_loss)
        print("\tCL Loss:", train_cl_loss)

        ###### Evaluation ######
        intervenable.model.eval()
        device = "cuda"
        # evaluation on the test set
        eval_labels = []
        eval_preds = []
        cl_hstates = []
        intrv_hstates = []
        test_actn_loss = 0.0
        test_cl_loss = 0.0
        with torch.no_grad():
            epoch_iterator = tqdm(test_dataloader, desc=f"Test")
            for step, inputs in enumerate(epoch_iterator):
                for k, v in inputs.items():
                    if v is not None and isinstance(v, torch.Tensor):
                        inputs[k] = v.to(device)
                b_s = inputs["input_ids"].shape[0]
                base_outputs, counterfactual_outputs = intervenable(
                    {"input_ids": inputs["input_ids"]},
                    [{"input_ids": inputs["source_input_ids"]}],
                    {"sources->base": 80},  # swap 80th token
                    output_original_output=True,
                    output_hidden_states=False,
                )
                eval_labels += [inputs["labels"]]
                eval_preds += [counterfactual_outputs.logits]
                intrv_hstates.append(comms_dict["intrv_vectors"].cpu())
                cl_hstates.append(test_cl_vector_dict[cl_keys[0]].cpu()[inputs["indices"].long().cpu()].cpu())

                test_actn_loss += calculate_loss(
                    counterfactual_outputs.logits,
                    inputs["labels"]
                ).item()/len(test_dataloader)
                test_cl_loss += calculate_cl_loss(
                    intrv_vectors=comms_dict["intrv_vectors"],
                    inputs=inputs,
                    cl_vector_dict=test_cl_vector_dict,
                    cl_failure_dict=test_cl_failure_dict,
                    das_object=das_object,
                    do_rotate=do_rotate,
                ).item()/len(test_dataloader)
        eval_metrics = compute_metrics(eval_preds, eval_labels)
        test_accuracy = eval_metrics["accuracy"]
        print("Test Acc:", test_accuracy)
        print("\tActn Loss:", test_actn_loss)
        print("\tCL Loss:", test_cl_loss)
        mask = das_object.get_boundary_mask(
            batch_size=1, subspaces=None, device=device, dtype=torch.float32
        )
        print("Mask:", mask.shape, "- Sum:", mask.sum())

        metrics_dict["test_accuracy"].append(test_accuracy)
        metrics_dict["test_actn_loss"].append(test_actn_loss)
        metrics_dict["test_cl_loss"].append(test_cl_loss)
    
        intrv_states = torch.vstack(intrv_hstates)
        natty_states = torch.vstack(cl_hstates)
    
        if save_actvs_dir is not None and not debug:
            actvs_name = f"{save_actvs_dir}/cl_eps_{cl_eps}_seed_{seed}_dorot{do_rotate}_{save_stamp}.pt"
            torch.save({
                "intrv_states": intrv_states,
                "natty_states": natty_states,
            }, actvs_name)
            print(f"Saved actvs to {actvs_name}")
    
        n_samples = n_divergence_samples
    
        print()
        print("Computing divergences for CL epsilon:", cl_eps)
        print("Natty:", natty_states.shape)
        print("Intrv:", intrv_states.shape)
        emd_df_dict = {
            "sample_id": [],
        }
        if debug: n_samples = 2
        for samp_id in range(n_samples):
            with torch.no_grad():
                if samp_id == 0:
                    visualize_states(
                        natty_states.cpu().detach().float(),
                        intrv_states.cpu().detach().float(),
                        xdim=0,
                        ydim=1,
                        save_name=f"figs/cl_das_divergence_{cl_eps}_seed_{seed}_dorot{do_rotate}_{save_stamp}.png" if not debug and samp_id == 0 else None,
                        visualize=True,
                        expl_var_threshold=0,
                        pca_batch_size=500,
                        use_numpy=use_numpy,
                    )
                natty_vecs = natty_states.cpu().detach().float()
                intrv_vecs = intrv_states.cpu().detach().float()
                diffs = collect_divergences(
                    natty_vecs, intrv_vecs, sample_size=5000)
            
            emd_df_dict["sample_id"].append(samp_id)
            for k,v in diffs.items():
                if k not in emd_df_dict:
                    emd_df_dict[k] = []
                emd_df_dict[k].append(float(v))

        emd_df = pd.DataFrame(emd_df_dict)
        print(emd_df)
        cols = list(diffs.keys())
        means = dict(emd_df[cols].mean())
        errors = dict(emd_df[cols].sem())
        print("Divergence means:")
        for col,val in means.items():
            if col not in metrics_dict:
                metrics_dict[col] = []
            if col+"_sem" not in metrics_dict:
                metrics_dict[col+"_sem"] = []
            metrics_dict[col].append(float(val))
            metrics_dict[col+"_sem"].append(float(errors[col]))
            print("\t", col, ":", float(val), "±", float(errors[col]), "sem")
        metrics_dict["method"].append("das")
        if not os.path.exists("csvs/"):
            os.mkdir("csvs/")
        emd_df = pd.DataFrame(metrics_dict)
        if not debug:
            csv_name = f"csvs/cl_das_divergences_dorot{do_rotate}_{save_stamp}.csv"
            emd_df.to_csv(csv_name, header=True, index=False)
            print(f"Saved divergences to {csv_name}")
        print()





