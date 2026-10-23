"""Reproduce all 73 reported accuracy points from manuscript Table VI.
Run: python plot_multipair.py [--out-dir figures]
Requirements: matplotlib. Missing configurations are not interpolated.
"""
import argparse
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

DATA = {
 'present80': ('PRESENT-80', {
 7: [(1,.9516),(2,.9919),(4,.9954),(8,.9988),(16,.9990),(32,.9920)],
 8: [(1,.6766),(2,.7556),(4,.8438),(8,.9235),(16,.9783),(32,.9822)],
 9: [(1,.5161),(2,.5396),(4,.5674),(8,.6170),(16,.6763),(32,.7552)],
 10: [(8,.4999),(16,.5356),(32,.5515)]}),
 'simeck3264': ('SIMECK-32/64', {
 12: [(1,.8787),(2,.9408),(4,.9883),(8,.9976),(16,.9987),(32,.9917)],
 13: [(1,.6754),(2,.7568),(4,.8098),(8,.8149),(16,.9425),(32,.9510)],
 14: [(2,.5662),(4,.5777),(8,.5875),(16,.6702),(32,.7810)],
 15: [(16,.5489),(32,.5933)]}),
 'hight': ('HIGHT', {
 12: [(1,.7355),(2,.8507),(4,.9195),(8,.9664),(16,.9911),(32,.9987)],
 13: [(1,.5798),(2,.5992),(4,.6361),(8,.6907),(16,.7572),(32,.8426)],
 14: [(16,.5820),(32,.6252)]}),
 'lea128': ('LEA-128', {
 7: [(1,.9244),(2,.9132),(4,.9320),(8,.9563),(16,.9624),(32,.9701)],
 8: [(1,.7598),(2,.8086),(4,.8644),(8,.8854),(16,.8942),(32,.9134)],
 9: [(1,.5780),(2,.6191),(4,.6310),(8,.7332),(16,.8046),(32,.8454)],
 10: [(32,.5009)]})}
COLORS = ['#0072B2','#D55E00','#009E73','#CC79A7']
MARKERS = ['o','s','^','D']
STYLES = ['-','--','-.',':']

def draw(ax, series):
    ax.axhline(.5, color='0.45', ls='--', lw=.8, zorder=1)
    for j,(r,points) in enumerate(series.items()):
        x,y = zip(*points)
        ax.plot(x,y,color=COLORS[j],marker=MARKERS[j],
                ls=STYLES[j] if len(points)>1 else 'None',
                lw=1.25,ms=4,label=fr'$r={r}$',zorder=3)
    ax.set_xscale('log',base=2)
    ax.set_xlim(.88,36)
    ax.set_ylim(.48,1.015)
    ticks=[1,2,4,8,16,32]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([str(t) for t in ticks]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_yticks([.5,.6,.7,.8,.9,1.0])
    ax.set_xlabel('Pairs per sample, $k$')
    ax.set_ylabel('Test accuracy')
    ax.grid(True,color='.9',lw=.55)
    ax.set_axisbelow(True)
    ax.tick_params(direction='out',length=3)
    for spine in ax.spines.values():
        spine.set_color('.5')
        spine.set_linewidth(.6)
    # Keep legends outside the data area to avoid concealing high accuracies.
    ax.legend(loc='lower center',bbox_to_anchor=(.5,1.015),ncol=4,
              frameon=False,fontsize=7.5,handlelength=1.7,
              columnspacing=.9,handletextpad=.35,borderaxespad=0)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',type=Path,default=Path(__file__).resolve().parent/'figures')
    args=parser.parse_args()
    args.out_dir.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],
                         'font.size':9,'axes.labelsize':9,'xtick.labelsize':8,
                         'ytick.labelsize':8,'pdf.fonttype':42,'ps.fonttype':42})
    for slug,(title,series) in DATA.items():
        fig,ax=plt.subplots(figsize=(3.4,2.7))
        fig.subplots_adjust(left=.17,right=.98,bottom=.19,top=.84)
        draw(ax,series)
        # Identical canvas and margins keep all four exported panels aligned.
        fig.savefig(args.out_dir/f'{slug}.pdf')
        fig.savefig(args.out_dir/f'{slug}.png',dpi=300)
        plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(7.1,6.0))
    fig.subplots_adjust(left=.09,right=.98,bottom=.10,top=.88,wspace=.30,hspace=.65)
    for i,(ax,(slug,(title,series))) in enumerate(zip(axes.flat,DATA.items())):
        draw(ax,series)
        ax.set_title(f'({chr(97+i)}) {title}',y=1.25,fontsize=10)
    fig.text(.5,.025,'Dashed gray line: random guessing (accuracy = 0.5)',ha='center',fontsize=9)
    fig.savefig(args.out_dir/'multipair_preview.png',dpi=200)
    plt.close(fig)
    with (args.out_dir/'table_vi.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['cipher','round','pairs','accuracy'])
        for title,series in DATA.values():
            for r,points in series.items():
                for k,a in points: writer.writerow([title,r,k,f'{a:.4f}'])
    print(f'Exported {sum(len(p) for _,s in DATA.values() for p in s.values())} data points to {args.out_dir}')

if __name__=='__main__':
    main()