import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from geomloss import SamplesLoss
from fca.projections import perform_eigen_pca


kwargs = { "loss": "sinkhorn", "p": 2, "blur": 0.05, }
loss_fn = SamplesLoss(**kwargs)

def compute_emd(X,Y):
    return loss_fn(X.float(),Y.float())

def filter_by_layer_and_position(natty_hstates, intrv_hstates, layer=None, pos=None):
    """
    natty_hstates: torch tensor (B,Layer,Pos,D)
    intrv_hstates: torch tensor (B,Layer,Pos,D)
    layer: int
    pos: int
    """
    d = natty_hstates.shape[-1]
    if layer is not None:
        natty_hstates = natty_hstates[:,layer]
        intrv_hstates = intrv_hstates[:,layer]
    if pos is None:
        natty_states = [natty_hstates[i].reshape(-1,d) for i in range(len(natty_hstates))]
        intrv_states = [intrv_hstates[i].reshape(-1,d) for i in range(len(natty_hstates))]
    else:
        natty_states = [natty_hstates[i][pos].reshape(-1,d) for i in range(len(natty_hstates))]
        intrv_states = [intrv_hstates[i][pos].reshape(-1,d) for i in range(len(natty_hstates))]
    natty_states = torch.vstack(natty_states)
    intrv_states = torch.vstack(intrv_states)
    return natty_states, intrv_states


def sample_emd(X,Y, sample_type="identity", normalize=False, sample_size=None):
    """
    X: torch tensor (B,D)
    Y: torch tensor (B,D)
    sample_type: str
        "identity": computes emd between X and Y
        "permute": permutes both X and Y
        "sample": samples from each X and Y with replacement
    normalize: bool
        if true, will normalize X and Y after the sampling.
    """
    
    if sample_size is None: sample_size = len(X)
    sample_size = min(sample_size, len(X))
    if sample_type=="permute":
        X = X[torch.randperm(len(X)).long()]
        Y = Y[torch.randperm(len(X)).long()]
    elif sample_type=="sample":
        X = X[torch.randint(0,len(X),(len(X),)).long()]
        Y = Y[torch.randint(0,len(Y),(len(X),)).long()]
    if normalize:
        X = (X-X.mean(0))/(X.std(0)+1e-5)
        Y = (Y-Y.mean(0))/(Y.std(0)+1e-5)
    X = X[:sample_size]
    Y = Y[:sample_size]
    return compute_emd(X, Y).item()

def interleave_scatter(
    natty_df,
    intrv_df,
    hue="count",
    title=None,
    samps_per_step=50,
    incl_legend=False,
    leg_loc=None,
    leg_bbox_anchor=None,
    leg_alpha=0.7,
    fontsize=25,
    titlesize=30,
    legendsize=25,
    ticksize=30,
    labelsize=25,
    save_name=None,
    *args, **kwargs,
):
    
    fig = plt.figure()
    ax = plt.gca()
    
    alpha = 0.8,
    intrv_palette = "bright"
    natty_palette = "bright"
    dark = 0.2,
    light = 0.85,
    rot = 0,
    
    intrv_cmap = sns.color_palette(intrv_palette)
    color = None
    if hue is None:
        color = sns.color_palette(intrv_palette)[0]
    sns.scatterplot(x="pc0", y="pc1", color=color, alpha=alpha, data=intrv_df, ax=ax, hue=hue, palette=intrv_cmap, edgecolor="none")
    
    native_cmap = sns.color_palette(natty_palette)
    color = None
    if hue is None:
        color = sns.color_palette(natty_palette)[1]
    sns.scatterplot(x="pc0", y="pc1", color=color, alpha=alpha, data=natty_df, ax=ax, hue=hue, palette=native_cmap, edgecolor="none")
        
    plt.xlabel("", fontsize=fontsize)
    plt.ylabel("", fontsize=fontsize)
    plt.xticks([], fontsize=fontsize)
    plt.yticks([], fontsize=fontsize)
    
    
    if not incl_legend:
        plt.legend().set_visible(False)
    else:
        # Legend handles: colored rectangles with labels
        color = sns.color_palette(natty_palette)[1]
        native_patch = mpatches.Patch(color=color, label="Native")
        color = sns.color_palette(intrv_palette)[0]
        intrv_patch = mpatches.Patch(color=color, label="Intervened")
        ax.legend(handles=[native_patch, intrv_patch], fontsize=legendsize, loc="upper right", bbox_to_anchor=(1.75,1))
    if save_name:
        plt.savefig(save_name, dpi=600, bbox_inches="tight")
    
    plt.show()

def visualize_states(
    natty_states,
    intrv_states,
    xdim=0,
    ydim=1,
    save_name=None,
    expl_var_threshold=0,
    sample_size=None,
    emd_sample_type="permute",
    sig_figs = 5,
    emd_sample_size=5000,
    normalize_emd=True,
    visualize=True,
    verbose=True,
    incl_legend=False,
):

    if sample_size:
        samp = torch.randperm(len(natty_states))[:sample_size].long()
        natty_states = natty_states[samp]
        intrv_states = intrv_states[samp]
        
    X = np.concatenate([
        natty_states.clone().detach().cpu().float().numpy(),
        intrv_states.clone().detach().cpu().float().numpy(),
    ], axis=0)
    train_X = X
    
    ret = perform_eigen_pca(
        X=torch.tensor(train_X),
        scale=True,
        center=True,
        transform_data=True,
        batch_size=2000,
    )
    vecs = ret["transformed_X"]
    
    natty_vecs = torch.tensor(vecs[:len(natty_states)])
    intrv_vecs = torch.tensor(vecs[len(natty_states):])
    expl_vars = torch.tensor(ret["proportion_expl_var"]).float()
    
    if xdim is None or ydim is None:
        diffs = ((natty_vecs-intrv_vecs)**2).mean(0)
        #diffs = (1-torch.nn.functional.cosine_similarity(natty_vecs.T,intrv_vecs.T).T)
        valids = expl_vars>=expl_var_threshold
        diffs[~valids] = 0
        topk = torch.topk(diffs, 2)
        xdim, ydim = topk.indices
        
    xdim = int(xdim)
    ydim = int(ydim)
    if verbose:
        print("x-dim:", xdim)
        print("y-dim:", ydim)
        print("Top Expl Vars:", expl_vars[:5])
    
        print("Vis Expl Vars:",
              ret["proportion_expl_var"][xdim],
              ret["proportion_expl_var"][ydim]
             )
    
    if visualize:
        natty_df = {
            "pc0": natty_vecs[:,xdim],
            "pc1": natty_vecs[:,ydim],
            "hue": np.ones_like(natty_vecs[:,ydim]),
        }
        intrv_df = {
            "pc0": intrv_vecs[:,xdim],
            "pc1": intrv_vecs[:,ydim],
            "hue": np.ones_like(natty_vecs[:,ydim]),
        }
        
        interleave_scatter(
            natty_df,
            intrv_df,
            hue=None,
            title="", #f"Neural Space {varb}",
            incl_legend=incl_legend,
            save_name=save_name,
        )
    
    mse = ((natty_vecs-intrv_vecs)**2).mean().item()
    emd = sample_emd(natty_vecs, intrv_vecs, sample_type=emd_sample_type, normalize=normalize_emd, sample_size=emd_sample_size)
    base_emd = sample_emd(natty_vecs, natty_vecs, sample_type=emd_sample_type, normalize=normalize_emd, sample_size=emd_sample_size)
    
    return {
        "mse": round(mse, sig_figs),
        "emd": round(emd, sig_figs),
        "base_emd": round(base_emd, sig_figs),
    }
    