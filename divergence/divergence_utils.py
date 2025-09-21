import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from geomloss import SamplesLoss
import fca


kwargs = { "loss": "sinkhorn", "p": 2, "blur": 0.05, }
loss_fn = SamplesLoss(**kwargs)

def compute_emd(X,Y):
    return loss_fn(X,Y)

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
        X = X[torch.randperm(sample_size).long()]
        Y = Y[torch.randperm(sample_size).long()]
    elif sample_type=="sample":
        X = X[torch.randint(0,len(X),(sample_size,)).long()]
        Y = Y[torch.randint(0,len(Y),(sample_size,)).long()]
    if normalize:
        X = (X-X.mean(0))/(X.std(0)+1e-5)
        Y = (Y-Y.mean(0))/(Y.std(0)+1e-5)
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
    #intrv_cmap = sns.cubehelix_palette(start=-.3, rot=rot, dark=dark, light=light, reverse=True, as_cmap=True)
    #color = None
    #if hue is None:
    #    color = intrv_cmap(0.7)
    ##intrv_cmap = sns.dark_palette("blue", as_cmap=True)
    intrv_cmap = sns.color_palette(intrv_palette)
    color = None
    if hue is None:
        color = sns.color_palette(intrv_palette)[0]
    sns.scatterplot(x="pc0", y="pc1", color=color, alpha=alpha, data=intrv_df, ax=ax, hue=hue, palette=intrv_cmap, edgecolor="none")
    
    #native_cmap = sns.cubehelix_palette(start=0.7, rot=rot, dark=dark, light=light, reverse=True, as_cmap=True)
    #if hue is None:
    #    color = native_cmap(0.7)
    ##native_cmap = sns.dark_palette("red", as_cmap=True)
    native_cmap = sns.color_palette(natty_palette)
    color = None
    if hue is None:
        color = sns.color_palette(natty_palette)[1]
    sns.scatterplot(x="pc0", y="pc1", color=color, alpha=alpha, data=natty_df, ax=ax, hue=hue, palette=native_cmap, edgecolor="none")
        
    ## y divider
    #ax.plot([0,0],[-1,5], "k--", alpha=0.5)
    ## x dividers
    #for i in y_values[:-1]:
    #    y = i+0.5
    #    ax.plot([-2,2],[y,y], "k--", alpha=0.5)
    #plt.xlim([-2,2])
    #plt.ylim([-0.75,4.75])
    
    plt.xlabel("", fontsize=fontsize)
    plt.ylabel("", fontsize=fontsize)
    plt.xticks([], fontsize=fontsize)
    plt.yticks([], fontsize=fontsize)
    
    # # Manually create colorbars / legend patches
    # native_cmap = sns.cubehelix_palette(start=0.7, rot=rot, dark=dark, light=light, reverse=True, as_cmap=True)
    # intrv_cmap = sns.cubehelix_palette(start=-.3, rot=rot, dark=dark, light=light, reverse=True, as_cmap=True)
    
    # Legend handles: colored rectangles with labels
    native_patch = mpatches.Patch(color=color, label="Native")
    intrv_patch = mpatches.Patch(color=color, label="Intervened")
    
    if not incl_legend:
        plt.legend().set_visible(False)
    else:
        ax.legend(handles=[native_patch, intrv_patch], fontsize=legendsize, loc="upper right", bbox_to_anchor=(1.75,1))
    if save_name:
        plt.savefig(save_name, dpi=600, bbox_inches="tight")
        print("Saved to", save_name)
    
    plt.show()

def visualize_states(
    natty_states,
    intrv_states,
    xdim=0,
    ydim=1,
    save_name=None,
    expl_var_threshold=0,
    sample_size=None,
    sample_type="permute",
    sig_figs = 5,
    emd_sample_size=5000,
):

    if sample_size:
        samp = torch.randperm(len(natty_states))[:sample_size].long()
        natty_states = natty_states[samp]
        intrv_states = intrv_states[samp]
    print("Natty PCA")
    X = np.concatenate([
        natty_states.clone().detach().cpu().float().numpy(),
        intrv_states.clone().detach().cpu().float().numpy(),
    ], axis=0)
    train_X = X
    
    #pca = PCA()
    #pca.fit(train_X)
    #vecs = pca.transform(X)
    ret = fca.projections.perform_eigen_pca(
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
    print("natty:", natty_vecs.shape)
    print("intrv:", intrv_vecs.shape)
    
    print("Top Expl Vars:", expl_vars[:5])
    
    if xdim is None or ydim is None:
        diffs = ((natty_vecs-intrv_vecs)**2).mean(0)
        #diffs = (1-torch.nn.functional.cosine_similarity(natty_vecs.T,intrv_vecs.T).T)
        valids = expl_vars>=expl_var_threshold
        diffs[~valids] = 0
        topk = torch.topk(diffs, 2)
        xdim, ydim = topk.indices
        
    xdim = int(xdim)
    ydim = int(ydim)
    print("X:", xdim, "Y:", ydim)
    
    print("Expl Vars:",
          ret["proportion_expl_var"][xdim],
          ret["proportion_expl_var"][ydim]
         )
    
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
        incl_legend=False,
        save_name=save_name,
    )
    
    mse = ((natty_vecs-intrv_vecs)**2).mean().item()
    emd = sample_emd(natty_vecs, intrv_vecs, sample_type=sample_type, normalize=False, sample_size=emd_sample_size)
    base_emd = sample_emd(natty_vecs, natty_vecs, sample_type=sample_type, normalize=False, sample_size=emd_sample_size)
    
    return {
        "mse": round(mse, sig_figs),
        "emd": round(emd, sig_figs),
        "base_emd": round(base_emd, sig_figs),
    }
    