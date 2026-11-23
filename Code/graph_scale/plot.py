"""Plot Fig. 3 from experiment trial results."""
import argparse
from pathlib import Path
from ba_experiment import load_trial_metrics, summarize_trial_metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm,ticker
ROOT=Path(__file__).resolve().parent
font=next((f for f in ['Times New Roman','Nimbus Roman','Liberation Serif'] if f in {v.name for v in fm.fontManager.ttflist}),'DejaVu Serif')
plt.rcParams.update({'font.family':font,'font.size':15,'axes.labelsize':17,'xtick.labelsize':16,'ytick.labelsize':16,'mathtext.fontset':'stix','pdf.fonttype':42})
METHODS=[('local',r'$\mathbf{Local}_{r}^{(1)}$','#F2AA3B','s','-'),('ikos_shuffle',r'$\mathbf{Shuffle}_{r}^{(1)}$','#28B9DF','^','-'),('decentral',r'$\mathbf{Local}_{r}^{(2)}$','#83CC91','o','-')]
def draw(ax,d,x,y):
 for key,label,col,marker,ls in METHODS:
  s=d[d.mechanism==key].sort_values(x)
  ax.plot(s[x],s[y],label=label,color=col,marker=marker,linestyle=ls,lw=2.2,ms=6.5)
 ax.set_yscale('log');ax.set_ylabel(y);ax.yaxis.set_major_locator(ticker.LogLocator(base=10,numticks=5));ax.yaxis.set_minor_formatter(ticker.NullFormatter());ax.grid(axis='y',which='major',ls='--',color='#d6d6d6',lw=.7)
 for sp in ax.spines.values():sp.set_color('#888888')
def finish(fig,ax,name,out,top=1):
 h,l=ax.get_legend_handles_labels();fig.legend(h,l,loc='upper center',ncol=3,fontsize=16,frameon=False,bbox_to_anchor=(.53,top))
 for ext in ['png','pdf','svg']:fig.savefig(out/(name+'.'+ext),dpi=220,bbox_inches='tight',pad_inches=.08)
 plt.close(fig)
def plot_scale(d, output):
 fig,axs=plt.subplots(2,2,figsize=(9,6.6));fig.subplots_adjust(left=.105,right=.99,bottom=.17,top=.89,wspace=.29,hspace=.70)
 for row,metric in enumerate(['MSE','MARE']):
  for col,eps in enumerate([.5,1]):
   ax=axs[row,col];draw(ax,d[d.epsilon==eps],'n',metric)
   ax.set_xlim(-2000,103000);ax.set_xticks([0,20000,40000,60000,80000,100000]);ax.set_xticklabels(['0','2','4','6','8','10']);xlabel=r'Number of vertices $n$ ($\times 10^4$)'
   # Keep matching metric panels on the same vertical scale.
   vals=d[metric];ax.set_ylim(vals.min()/1.35,vals.max()*1.35)
   ax.set_xlabel(xlabel,labelpad=2);ax.annotate(f'({chr(97+row*2+col)}) '+r'$\varepsilon='+str(eps)+'$',xy=(.5,0),xycoords='axes fraction',xytext=(0,-50),textcoords='offset points',ha='center',va='top',fontsize=15,annotation_clip=False)
 finish(fig,axs[0,0],'scale_mse_mare',output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials', type=Path, required=True, help='Scale run output directory')
    parser.add_argument('--output', type=Path, default=ROOT/'figures')
    parser.add_argument('--save-data', type=Path, help='Also write the plotting table to this CSV')
    parser.add_argument('--data-only', action='store_true', help='Export the table without drawing')
    args = parser.parse_args()
    if args.data_only and not args.save_data:
        parser.error('--data-only requires --save-data')
    keys = ['n', 'ba_m', 'epsilon', 'mechanism']
    data = summarize_trial_metrics(load_trial_metrics(args.trials), keys).drop(columns='runs')
    if not args.data_only:
        args.output.mkdir(parents=True, exist_ok=True)
        plot_scale(data, args.output)
    if args.save_data:
        args.save_data.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(args.save_data, index=False)


if __name__ == '__main__':
    main()
