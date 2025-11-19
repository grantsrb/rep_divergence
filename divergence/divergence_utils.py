import math
import os
from tqdm import tqdm

import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from geomloss import SamplesLoss
from scipy.optimize import linear_sum_assignment
from scipy.stats import sem


from sklearn.neighbors import NearestNeighbors, KernelDensity
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM

def device_fn(device):
    if device==-1:
        return "cpu"
    return device

kwargs = { "loss": "sinkhorn", "p": 2, "blur": 0.05, }
loss_fn = SamplesLoss(**kwargs)

def compute_emd(X,Y):
    return loss_fn(X.float(),Y.float())


###############################################################
# Rotation Matrices
###############################################################
class PositiveSymmetricDefiniteMatrix(torch.nn.Module):
    def __init__(self, size, identity_init=False, *args, **kwargs):
        super().__init__()
        self.eps = 1e-1
        self.size = size
        self.core_mtx = torch.nn.Parameter(
            torch.randn(size,size)/math.sqrt(size))
        if identity_init:
            self.core_mtx.data = torch.eye(size)

    def get_psd_mtx(self):
        return torch.mm(self.core_mtx, self.core_mtx.T) +\
            self.eps*torch.eye(
                self.core_mtx.shape[-1],
                device=device_fn(self.core_mtx.get_device()),
            )

    @property
    def weight(self):
        return self.get_psd_mtx()

    def inv(self):
        """
        Computes the inverse of a positive symmetric-definite matrix using Cholesky
        decomposition.
        """
        L = torch.linalg.cholesky(self.weight)
        return torch.cholesky_inverse(L)

class SymmetricDefiniteMatrix(PositiveSymmetricDefiniteMatrix):
    """
    Similar to a PSD matrix, but learns signs to multiply rows of the
    PSD matrix to allow it to be negative
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.signs = torch.nn.Parameter(0.01*torch.randn(self.size))

    @property
    def weight(self):
        psd = self.get_psd_mtx()
        signs = torch.nn.functional.tanh(self.signs)
        signs = signs + self.eps*torch.sign(signs) # offset to ensure nonzero
        return psd*signs

    def inv(self):
        """
        Computes the inverse of a positive symmetric-definite matrix using Cholesky
        decomposition.
        """
        return torch.linalg.inv(self.weight)



###############################################################
# Data Manipulation and Plotting
###############################################################

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
    emd = compute_emd(X, Y).item()
    if normalize:
        emd = emd/(X.shape[-1]**0.5)
    return emd

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
        try:
            plt.savefig(save_name, dpi=600, bbox_inches="tight")
        except:
            print(f"Error saving figure to {save_name}")
            os.makedirs(os.path.dirname(save_name), exist_ok=True)
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
    visualize=True,
    verbose=True,
    incl_legend=False,
    pca_batch_size=1000,
    use_numpy=False,
):

    if sample_size:
        samp = torch.randperm(len(natty_states))[:sample_size].long()
        natty_states = natty_states[samp]
        intrv_states = intrv_states[samp]
        
    if use_numpy:
        X = np.concatenate([
            natty_states.clone().detach().cpu().float().numpy(),
            intrv_states.clone().detach().cpu().float().numpy(),
        ], axis=0)
    else:
        X = torch.cat([natty_states, intrv_states], dim=0)
    train_X = X
    
    ret = perform_eigen_pca(
        #X=torch.tensor(train_X),
        X=train_X,
        scale=True,
        center=True,
        transform_data=True,
        batch_size=pca_batch_size,
    )
    vecs = ret["transformed_X"]
    
    natty_vecs = torch.tensor(vecs[:len(natty_states)]).cpu().float()
    intrv_vecs = torch.tensor(vecs[len(natty_states):]).cpu().float()
    expl_vars = torch.tensor(ret["proportion_expl_var"]).float().cpu()
    
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

def collect_divergences(
    natty_vecs,
    intrv_vecs,
    sample_size=5000,
):
    if sample_size:
        samp = torch.randperm(len(natty_vecs))[:sample_size].long()
        natty_vecs = natty_vecs[samp]
        intrv_vecs = intrv_vecs[samp]
        
    mse = ((natty_vecs-intrv_vecs)**2).mean().item()

    half = len(natty_vecs)//2
    perm = torch.randperm(len(natty_vecs)).long()
    nat = natty_vecs[perm[:half]]
    intrv = intrv_vecs[perm[half:]]
    div_dict = divergences(nat, intrv)
    intrv = natty_vecs[perm[half:]]
    div_dict2 = divergences(nat, intrv)
    return {
        "mse": mse,
        **div_dict,
        **{"base_"+k:v for k,v in div_dict2.items()}
    }

    

def divergences(
        natty_vecs,
        intrv_vecs,
        emd_sample_type="permute",
        normalize_emd=True,
        emd_sample_size=5000,
):
    div_dict = {}

    div_dict["emd"] = sample_emd(
        natty_vecs,
        intrv_vecs,
        sample_type=emd_sample_type,
        normalize=normalize_emd,
        sample_size=emd_sample_size
    )

    cost_mtx = 1-get_cor_mtx(intrv_vecs.T, natty_vecs.T, zscore=True, to_numpy=True) # half x half
    _, div_dict["cost_cos"] = optimal_pairs(cost_mtx)
    
    cost_mtx = get_mse_mtx(intrv_vecs, natty_vecs, to_numpy=True) # half x half
    _, div_dict["cost_mse"] = optimal_pairs(cost_mtx)

    local_pca = LocalPCADistance(natty_vecs.cpu().float().numpy())
    div_dict["local_pca"] = local_pca.score(
        intrv_vecs.cpu().float().numpy(), verbose=True).mean()
    
    local_recon = LLEReconstructionDistance(natty_vecs.cpu().float().numpy())
    div_dict["lle_recon"] = local_recon.score(
        intrv_vecs.cpu().float().numpy(), verbose=True).mean()

    kde = KDEDensityScore(natty_vecs.cpu().float().numpy(), bandwidth=0.4)
    div_dict["kde"] = kde.score(
        intrv_vecs.cpu().float().numpy(), verbose=True).mean()
    
    svm = OneClassSVMDistance(natty_vecs.cpu().float().numpy())
    div_dict["svm"] = svm.score(
        intrv_vecs.cpu().float().numpy(), verbose=True).mean()
    
    return {k:float(v) for k,v in div_dict.items()}

###############################################################
# Linear Algebra Utilities
###############################################################

def matrix_projinv(x, W):
    """
    This function projects the activations into the weight space and then
    inverts the projection, returning the inverted vectors.
    
    Args:
        x: torch tensor (B,D)
        W: torch tensor (D,P)
    Returns:
        torch tensor (B,D)
            the projected activations returned to their
            original space.
    """
    return torch.matmul(torch.matmul(x, W), torch.linalg.pinv(W))

def explained_variance(
    preds: torch.Tensor,
    labels:torch.Tensor,
    eps: float = 1e-8,
    mean_over_dims=False,
) -> torch.Tensor:
    """
    Caculates the explained variance of the reps on the
    target reps.
    
    Args:
        preds: torch tensor (B,D)
        labels: torch tensor (B,D)
        eps: float
            small constant to prevent division by zero
        mean_over_dims: bool
            if true, will return the mean explained variance over the
            feature dimensions
    Returns:
        expl_var: torch tensor (D,) or (1,)
            we get an explained variance value for each dimension, but
            we can average over this value if mean_over_dims is true.
    """
    assert preds.shape == labels.shape, "Shapes of preds and labels must match"
    
    diff = labels - preds
    var_diff = torch.var(diff, dim=0, unbiased=True)    # shape (D,)
    var_labels = torch.var(labels, dim=0, unbiased=True)# shape (D,)

    expl_var = 1 - var_diff / (var_labels + eps)         # shape (D,)
    if mean_over_dims:
        return expl_var.mean()
    return expl_var

def projinv_expl_variance(x,W):
    """
    Projects x into W and inverts the projection to create z. Then
    returns the explained variance of x using z.

    Args:
        x: torch tensor (B,D)
        W: torch tensor (D,P)
    Returns:
        explained_variance: torch tensor (1,)
            the explained variance of the activations projected into W
            and then returned to their original space.
    """
    preds = matrix_projinv(x,W)
    return explained_variance(preds, x)

def lost_variance(x,W):
    """
    Returns the variance lost when x is right multiplied by W.

    Args:
        x: torch tensor (B,D)
        W: torch tensor (D,P)
    Returns:
        lost_var: torch tensor (1,)
            1 minus the explained variance of x projected into W
            and then returned to their original space.
    """
    return 1-projinv_expl_variance(x,W)

def component_wise_expl_var(actvs, weight, eps=1e-6):
    """
    For each component in U, will determine its projected
    explained variance.
    
    Args:
        actvs: torch tensor (B,D)
        weight: torch tensor (D,P)
        eps: float
            a value for which components will be removed
    Returns:
        expl_vars: tensor (P,)
            the explained variance for each component
        cumu_expl_vars: tensor (P,)
            the explained variance for the cumulation of the components
    """
    U, S, Vt = torch.linalg.svd(weight)
    n_components = (S>=eps).long().sum()
    expl_vars = []
    cumu_sum = 0
    cumu_expl_vars = []
    for comp in range(n_components):
        W = S[comp]*torch.matmul(U[:,comp:comp+1], Vt[comp:comp+1, :])
        preds = matrix_projinv(actvs, W=W)
        expl_var = explained_variance(preds, actvs)
        expl_vars.append(expl_var)

        cumu_sum += preds
        expl_var =  explained_variance(cumu_sum, actvs)
        cumu_expl_vars.append(expl_var)
    for _ in range(max(*weight.shape)-n_components):
        expl_vars.append(torch.zeros_like(expl_vars[-1]))
        cumu_expl_vars.append(torch.zeros_like(cumu_expl_vars[-1]))
    return torch.stack(expl_vars), torch.stack(cumu_expl_vars)

def get_cor_mtx(X, Y, batch_size=500, to_numpy=False, zscore=True, device=None):
    """
    Creates a correlation matrix for X and Y using the GPU

    X: torch tensor or ndarray (B, C) or (B, C, H, W)
    Y: torch tensor or ndarray (B, K) or (B, K, H1, W1)
    batch_size: int
        batches the calculation if this is not None
    to_numpy: bool
        if true, returns matrix as ndarray
    zscore: bool
        if true, both X and Y are normalized over the T dimension
    device: int
        optionally argue a device to use for the matrix multiplications

    Returns:
        cor_mtx: (C,K) or (C*H*W, K*H1*W1)
            the correlation matrix
    """
    if len(X.shape) < 2:
        X = X[:,None]
    if len(Y.shape) < 2:
        Y = Y[:,None]
    if len(X.shape) > 2:
        X = X.reshape(len(X), -1)
    if len(Y.shape) > 2:
        Y = Y.reshape(len(Y), -1)
    if type(X) == type(np.array([])):
        to_numpy = True
        X = torch.FloatTensor(X)
        Y = torch.FloatTensor(Y)
    if device is None:
        device = X.get_device()
        if device<0: device = "cpu"
    if zscore:
        xmean = X.mean(0)
        xstd = torch.sqrt(((X-xmean)**2).mean(0))
        ymean = Y.mean(0)
        ystd = torch.sqrt(((Y-ymean)**2).mean(0))
        xstd[xstd<=0] = 1
        X = (X-xmean)/(xstd+1e-5)
        ystd[ystd<=0] = 1
        Y = (Y-ymean)/(ystd+1e-5)
    X = X.permute(1,0)

    with torch.no_grad():
        if batch_size is None:
            X = X.to(device)
            Y = Y.to(device)
            cor_mtx = torch.einsum("it,tj->ij", X, Y).detach().cpu()
        else:
            cor_mtx = []
            for i in range(0,len(X),batch_size): # loop over x neurons
                sub_mtx = []
                x = X[i:i+batch_size].to(device)

                # Loop over y neurons
                for j in range(0,Y.shape[1], batch_size):
                    y = Y[:,j:j+batch_size].to(device)
                    cor_block = torch.einsum("it,tj->ij",x,y)
                    cor_block = cor_block.detach().cpu()
                    sub_mtx.append(cor_block)
                cor_mtx.append(torch.cat(sub_mtx,dim=1))
            cor_mtx = torch.cat(cor_mtx, dim=0)
    cor_mtx = cor_mtx/len(Y)
    if to_numpy:
        return cor_mtx.numpy()
    return cor_mtx

def get_mse_mtx(X,Y, to_numpy=False):
    """
    Args:
        X: torch tensor (B,D)
        Y: torch tensor (N,D)
    
    Returns
        M: torch tensor (B,N)
    """
    M = []
    for samp in range(len(X)):
        M.append(((Y - X[samp])**2).mean(-1))
    if type(X)==np.ndarray:
        M = np.vstack(M)
    else:
        M = torch.vstack(M)
    if to_numpy:
        M = M.cpu().data.numpy()
    return M

def optimal_pairs(cost: np.ndarray):
    # cost: (m, m) matrix of pairwise distances
    row_ind, col_ind = linear_sum_assignment(cost)  # Hungarian under the hood
    total_cost = cost[row_ind, col_ind].sum()
    pairs = list(zip(row_ind.tolist(), col_ind.tolist()))
    return pairs, float(total_cost)

def sample_without_replacement(mtx, n_samples):
    perm = torch.randperm(len(mtx)).long()
    return mtx[perm[:n_samples]]

def perform_pca(
        X,
        n_components=None,
        scale=True,
        center=True,
        transform_data=False,
        full_matrices=False,
        randomized=False,
        use_eigen=True,
        batch_size=None,
        verbose=True,
):
    """
    Perform PCA on the data matrix X

    Args:
        X: tensor (M,N)
        n_components: int
            optionally specify the number of components
        scale: bool
            if true, will scale the data along each column
        transform_data: bool
            if true, will compute and return the transformed
            data
        full_matrices: bool
            determines if U will be returned as a square.
        randomized: bool
            if true, will use randomized svd for faster
            computations
        use_eigen: bool
            if true, will use an eigen decomposition on the
            covariance matrix of X to save compute
        batch_size: int or None
            optionally argue a batch size. only applies if use_eigen
            is true.
    Returns:
        ret_dict: dict
            A dictionary containing the following keys:
            - "components": tensor (N, n_components)
                The principal components (eigenvectors) of the data.
            - "explained_variance": tensor (n_components,)
                The explained variance for each principal component.
            - "proportion_expl_var": tensor (n_components,)
                The proportion of explained variance for each principal component.
            - "means": tensor (N,)
                The mean of each feature (column) in the data.
            - "stds": tensor (N,)
                The standard deviation of each feature (column) in the data.
            - "transformed_X": tensor (M, n_components)
                The data projected onto the principal components, if
                transform_data is True.
    """
    if use_eigen:
        return perform_eigen_pca(
            X=X,
            n_components=n_components,
            scale=scale,
            center=center,
            transform_data=transform_data,
            batch_size=batch_size,
            verbose=verbose,
        )
    if n_components is None:
        n_components = X.shape[-1]
        
    svd_kwargs = {}
    if type(X)==torch.Tensor:
        if randomized:
            svd_kwargs["q"] = n_components
            svd = torch.svd_lowrank
        else:
            svd_kwargs["full_matrices"] = full_matrices
            svd = torch.linalg.svd
    elif type(X)==np.ndarray:
        if randomized:
            svd_kwargs["n_components"] = n_components
            svd = np.linalg.svd_lowrank
        else:
            svd_kwargs["n_components"] = n_components
            svd_kwargs["compute_uv"] = True
            svd = np.linalg.svd
    assert not n_components or X.shape[-1]>=n_components

    # Center the data by subtracting the mean along each feature (column)
    means = torch.zeros_like(X[0])
    if center:
        means = X.mean(dim=0, keepdim=True)
        X = X - means
    stds = torch.ones_like(X[0])
    if scale:
        stds = (X.std(0)+1e-6)
        X = X/stds
    
    
    # Compute the SVD of the centered data
    # X = U @ diag(S) @ Vh, where Vh contains the principal components as its rows
    if verbose: print("Performing SVD")
    U, S, Vh = svd(X, **svd_kwargs)
    
    # The principal components (eigenvectors) are the first n_components rows of Vh
    components = Vh[:n_components]
    
    # Explained variance for each component can be computed from the singular values
    explained_variance = (S[:n_components] ** 2) / (X.shape[0] - 1)
    proportion_expl_var = explained_variance/explained_variance.sum()
    
    ret_dict = {
        "components": components,
        "explained_variance": explained_variance,
        "proportion_expl_var": proportion_expl_var,
        "means": means,
        "stds": stds,
    }
    if transform_data:
        # Project the data onto the principal components
        # Note: components.T has shape (features, n_components)
        ret_dict["transformed_X"] = X @ components.T

    return ret_dict

def perform_eigen_pca(
        X,
        n_components=None,
        scale=True,
        center=True,
        transform_data=False,
        batch_size=None,
        as_numpy=False,
        verbose=True,
):
    """
    Perform PCA on the data matrix X by using an eigen decomp on
    the covariance matrix

    Args:
        X: tensor (M,N)
        n_components: int
            optionally specify the number of components
        scale: bool
            if true, will scale the data along each column
        transform_data: bool
            if true, will compute and return the transformed
            data
    Returns:
        ret_dict: dict
            A dictionary containing the following keys:
            - "components": tensor (N, n_components)
                The principal components (eigenvectors) of the data.
            - "explained_variance": tensor (n_components,)
                The explained variance for each principal component.
            - "proportion_expl_var": tensor (n_components,)
                The proportion of explained variance for each principal component.
            - "means": tensor (N,)
                The mean of each feature (column) in the data.
            - "stds": tensor (N,)
                The standard deviation of each feature (column) in the data.
            - "transformed_X": tensor (M, n_components)
                The data projected onto the principal components, if
                transform_data is True.
    """
    if n_components is None:
        n_components = X.shape[-1]
        
    if type(X)==torch.Tensor:
        eigen_fn = torch.linalg.eigh
    elif type(X)==np.ndarray:
        eigen_fn = np.linalg.eigh
    assert not n_components or X.shape[-1]>=n_components

    # Center the data by subtracting the mean along each feature (column)
    means = 0
    if center:
        means = X.mean(0)
        X = X - means
    stds = 1
    if scale:
        stds = (X.std(0)+1e-6)
        X = X/stds
    
    cov = get_cor_mtx( # features x features shape (N,N)
        X,X,
        zscore=False,
        batch_size=batch_size
    )
    ## Use eigendecomposition of the covariance matrix for efficiency
    ## Cov = (1 / (M - 1)) * X^T X
    #cov = X.T @ X / (X.shape[0] - 1)  # shape (N, N)

    # Compute eigenvalues and eigenvectors
    eigvals, eigvecs = eigen_fn(cov)  # eigvals in ascending order
    if type(eigvals)==np.ndarray:
        eigvals = torch.tensor(eigvals)
        eigvecs = torch.tensor(eigvecs)

    # Select top n_components in descending order
    eigvals = eigvals[-n_components:].flip(0)
    eigvecs = eigvecs[:, -n_components:].flip(1)  # shape (N, n_components)

    explained_variance = eigvals
    proportion_expl_var = explained_variance / explained_variance.sum()
    components = eigvecs.T  # shape (n_components, N)

    ret_dict = {
        "components": components,
        "explained_variance": explained_variance,
        "proportion_expl_var": proportion_expl_var,
        "means": torch.tensor(means),
        "stds": torch.tensor(stds),
    }
    if transform_data:
        # Project the data onto the principal components
        # Note: components.T has shape (features, n_components)
        if type(X)==np.ndarray:
            X = torch.tensor(X)
        ret_dict["transformed_X"] = X @ components.T.float()
    
    if as_numpy:
        ret_dict = {k: v.cpu().detach().numpy() for k, v in ret_dict.items()}

    return ret_dict


class LocalPCADistance:
    def __init__(self, X, k=20, d=None):
        """
        X: (n, D) reference point cloud (the manifold samples)
        k: neighbors used to estimate local tangent
        d: intrinsic dim estimate (if None, keep components explaining 95% var)
        """
        self.X = np.asarray(X, float)
        self.nn = NearestNeighbors(n_neighbors=k).fit(self.X)
        self.k = k
        self.d = d

    def score(self, vecs, pca_expl_var=0.95, verbose=False):
        """
        Returns: residual norm (Euclidean) to the locally linear manifold at x.
        Larger = more off-manifold.
        """
        if len(vecs.shape)<=1: vecs = vecs[None]
        res_norms = []
        rng = range(len(vecs))
        if verbose:
            rng = tqdm(rng)
        for samp_idx in rng:
            x = np.asarray(vecs[samp_idx], float).reshape(1, -1)
            idx = self.nn.kneighbors(x, return_distance=False)[0]
            Xk = self.X[idx]
            xc = Xk - Xk.mean(0, keepdims=True)
    
            pca = PCA().fit(xc)
            if self.d is None:
                # Keep the smallest d achieving 95% variance (at least 1)
                cum = np.cumsum(pca.explained_variance_ratio_)
                d = max(1, int(np.searchsorted(cum, pca_expl_var) + 1))
            else:
                d = self.d
    
            U = pca.components_[:d]                # (d, D)
            proj = (x - Xk.mean(0))[0] @ U.T @ U   # project onto local subspace
            resid = (x - Xk.mean(0))[0] - proj
            res_norm = float(np.linalg.norm(resid))
            res_norms.append(res_norm)
        return np.asarray(res_norms)


class LLEReconstructionDistance:
    def __init__(self, X, k=20, reg=1e-3):
        self.X = np.asarray(X, float)
        self.nn = NearestNeighbors(n_neighbors=k).fit(self.X)
        self.k = k
        self.reg = reg

    def score(self, vecs, verbose=False):
        if len(vecs.shape)<=1: vecs = vecs[None]
        recons = []
        rng = range(len(vecs))
        if verbose:
            rng = tqdm(rng)
        for samp_idx in rng:
            x = np.asarray(vecs[samp_idx], float)
            idx = self.nn.kneighbors([x], return_distance=False)[0]
            Xi = self.X[idx]                  # (k, D)
            Zi = Xi - x                       # neighbor diffs
            C = Zi @ Zi.T                     # (k, k)
            C.flat[::C.shape[0]+1] += self.reg * np.trace(C) / self.k  # reg on diag
            # Solve C w = 1, then normalize so sum w = 1 (Roweis & Saul 2000)
            ones = np.ones(self.k)
            w = np.linalg.solve(C, ones)
            w /= w.sum()
            recon = (w @ Xi)
            recons.append(float(np.linalg.norm(x - recon)))
        return np.asarray(recons)

class KDEDensityScore:
    def __init__(self, X, bandwidth=0.5, kernel="gaussian"):
        self.kde = KernelDensity(bandwidth=bandwidth, kernel=kernel).fit(X)

    def score(self, vecs, verbose=False, *args, **kwargs):
        if len(vecs.shape)<=1: vecs = vecs[None]
        neg_logps = []
        rng = range(len(vecs))
        if verbose:
            rng = tqdm(rng)
        for samp_idx in rng:
            # Higher value = *more* typical; invert for an off-manifold score
            logp = float(self.kde.score_samples(np.asarray(vecs[samp_idx], float).reshape(1, -1))[0])
            neg_logps.append(-logp)
        return np.asarray(neg_logps)  # larger = more off-manifold

class OneClassSVMDistance:
    def __init__(self, X, nu=0.05, gamma="scale"):
        self.svm = OneClassSVM(nu=nu, kernel="rbf", gamma=gamma).fit(X)

    def score(self, vecs, verbose=False):
        if len(vecs.shape)<=1: vecs = vecs[None]
        scores = []
        rng = range(len(vecs))
        if verbose:
            rng = tqdm(rng)
        for samp_idx in rng:
            # More negative = more off-manifold; flip sign to make larger = more off
            v = vecs[samp_idx]
            arr = np.asarray(vecs[samp_idx], float).reshape(1, -1)
            score = -self.svm.decision_function(arr)[0]
            scores.append(float(score))
        return np.asarray(scores)


__all__ = [
    "matrix_projinv", "projinv_expl_variance", "lost_variance", "explained_variance",
    "component_wise_expl_var", "perform_pca",
]

if __name__=="__main__":
    n_dims = 4
    U,_,Vt = torch.linalg.svd(torch.randn(n_dims,n_dims), full_matrices=True)
    indys, cumu = component_wise_expl_var(U,U)
    print("Indy Components:", indys.mean(-1))
    print("\tSum:", indys.mean(-1).sum(0))
    print("Cumu Components:", cumu.mean(-1))
    preds = matrix_projinv(U,U)
    assert cumu.mean(-1)[-1]==explained_variance(preds, U).mean()
    assert explained_variance(preds, U).mean()==projinv_expl_variance(U,U).mean()
    assert lost_variance(U, U).mean()==(1-projinv_expl_variance(U,U).mean())