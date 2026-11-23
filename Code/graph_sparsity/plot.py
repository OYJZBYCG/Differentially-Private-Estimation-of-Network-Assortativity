"""Plot Fig. 4 from BA and real graph trial results."""
from pathlib import Path
import argparse
import pandas as pd
from ba_experiment import load_trial_metrics, summarize_trial_metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm,ticker
ROOT=Path(__file__).resolve().parent
font=next((f for f in ['Times New Roman','Nimbus Roman','Liberation Serif'] if f in {v.name for v in fm.fontManager.ttflist}),'DejaVu Serif')
plt.rcParams.update({'font.family':font,'font.size':15,'axes.labelsize':17,'xtick.labelsize':16,'ytick.labelsize':16,'mathtext.fontset':'stix','pdf.fonttype':42})
methods=[('local',r'$\mathbf{Local}_{r}^{(1)}$','#F2AA3B','s','-'),('ikos_shuffle',r'$\mathbf{Shuffle}_{r}^{(1)}$','#28B9DF','^','-'),('decentral',r'$\mathbf{Local}_{r}^{(2)}$','#83CC91','o','-')]
def plot_sparsity(d, output):
    fig,axs=plt.subplots(3,4,figsize=(18,9.6));fig.subplots_adjust(left=.055,right=.995,bottom=.11,top=.91,wspace=.34,hspace=.69)
    for row,name in enumerate(['BA','Deezer','GitHub']):
     f=d[d.dataset==name]
     for col,(metric,eps) in enumerate([('MSE',.5),('MSE',1),('MARE',.5),('MARE',1)]):
      ax=axs[row,col]
      for key,label,color,marker,ls in methods:
       s=f[(f.epsilon==eps)&(f.mechanism==key)].sort_values('mean_degree')
       expected=8 if name=='BA' else 7
       if len(s)!=expected or s[metric].isna().any() or (s[metric]<=0).any():raise ValueError(f'Invalid plotting data: {name}/{metric}/{eps}/{key}')
       ax.plot(s.mean_degree,s[metric],label=label,color=color,marker=marker,linestyle=ls,lw=2.2,ms=6.5)
      ax.set_yscale('log');ax.set_ylabel(metric);vals=f[metric];ax.set_ylim(vals.min()/1.35,vals.max()*1.35)
      if name=='BA':ax.set_xlim(-3,203);ax.set_xticks([0,40,80,120,160,200])
      elif name=='Deezer':ax.set_xlim(0,7);ax.set_xticks([0,2,4,6])
      else:ax.set_xlim(0,16);ax.set_xticks([0,4,8,12,16])
      ax.set_xlabel(r'Average degree $\overline{d}$',labelpad=2)
      ax.yaxis.set_major_locator(ticker.LogLocator(base=10,numticks=5));ax.yaxis.set_minor_formatter(ticker.NullFormatter());ax.grid(axis='y',which='major',ls='--',color='#d6d6d6',lw=.7)
      for spine in ax.spines.values():spine.set_color('#888888')
      ax.annotate(f'({chr(97+row*4+col)}) '+name+r', $\varepsilon='+f'{eps:g}'+'$',xy=(.5,0),xycoords='axes fraction',xytext=(0,-50),textcoords='offset points',ha='center',va='top',fontsize=16,annotation_clip=False)
    h,l=axs[0,0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',ncol=3,fontsize=16,frameon=False,bbox_to_anchor=(.53,.99))
    for ext in ['png','pdf','svg']:fig.savefig(output/('combined_sparsity_3x4.'+ext),dpi=220,bbox_inches='tight',pad_inches=.08)
    print('Font:',font,'; figures:',output.resolve())
    


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ba-trials', type=Path, required=True, help='BA run output directory')
    parser.add_argument('--real-trials', type=Path, nargs='+', required=True,
                        help='Real graph run output directories')
    parser.add_argument('--output', type=Path, default=ROOT/'figures')
    parser.add_argument('--save-data', type=Path, help='Also write the plotting table to this CSV')
    parser.add_argument('--data-only', action='store_true', help='Export the table without drawing')
    args = parser.parse_args()
    if args.data_only and not args.save_data:
        parser.error('--data-only requires --save-data')
    frames = [load_trial_metrics(args.ba_trials).assign(setting=lambda d: d.ba_m)]
    frames.extend(load_trial_metrics(folder).assign(setting=lambda d: d.retain_probability)
                  for folder in args.real_trials)
    data = summarize_trial_metrics(pd.concat(frames, ignore_index=True),
                                   ['dataset', 'setting', 'epsilon', 'mechanism'])
    if not args.data_only:
        args.output.mkdir(parents=True, exist_ok=True)
        plot_sparsity(data, args.output)
    if args.save_data:
        args.save_data.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(args.save_data, index=False)


if __name__ == '__main__':
    main()
