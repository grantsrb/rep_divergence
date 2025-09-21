import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns

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
    #df = pd.DataFrame({
    #    "x": samples[:,0],
    #    "y": samples[:,1],
    #    "hue": samples[:,1],
    #})
    #df["x"] = (df["x"]-np.mean(df["x"]))
    #df["hue"] = df["hue"]-np.min(df["hue"])
    #df["hue"] = df["hue"]/np.max(df["hue"])
    
    
    
    fig = plt.figure()
    ax = plt.gca()
    
    alpha = 0.8,
    dark = 0.2,
    light = 0.85,
    rot = 0,
    intrv_cmap = sns.cubehelix_palette(start=-.3, rot=rot, dark=dark, light=light, reverse=True, as_cmap=True)
    color = None
    if hue is None:
        color = intrv_cmap(0.7)
    #intrv_cmap = sns.dark_palette("blue", as_cmap=True)
    sns.scatterplot(x="pc0", y="pc1", color=color, alpha=alpha, data=intrv_df, ax=ax, hue=hue, palette=intrv_cmap, edgecolor="none")
    
    native_cmap = sns.cubehelix_palette(start=0.7, rot=rot, dark=dark, light=light, reverse=True, as_cmap=True)
    if hue is None:
        color = native_cmap(0.7)
    #native_cmap = sns.dark_palette("red", as_cmap=True)
    sns.scatterplot(x="pc0", y="pc1", color=color, alpha=alpha, data=natty_df, ax=ax, hue=hue, palette=native_cmap, edgecolor="none")
                    #hue="hue", palette="blue")
        
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
    
    #plt.xticks([], fontsize=fontsize)
    #plt.yticks([], fontsize=fontsize)
    
    # # Manually create colorbars / legend patches
    # native_cmap = sns.cubehelix_palette(start=0.7, rot=rot, dark=dark, light=light, reverse=True, as_cmap=True)
    # intrv_cmap = sns.cubehelix_palette(start=-.3, rot=rot, dark=dark, light=light, reverse=True, as_cmap=True)
    
    # Legend handles: colored rectangles with labels
    native_patch = mpatches.Patch(color=native_cmap(0.8), label="Native")
    intrv_patch = mpatches.Patch(color=intrv_cmap(0.8), label="Intervened")
    
    if not incl_legend:
        plt.legend().set_visible(False)
    else:
        ax.legend(handles=[native_patch, intrv_patch], fontsize=legendsize, loc="upper right", bbox_to_anchor=(1.75,1))
    if save_name:
        plt.savefig(save_name, dpi=600, bbox_inches="tight")
        print("Saved to", save_name)
    
    plt.show()