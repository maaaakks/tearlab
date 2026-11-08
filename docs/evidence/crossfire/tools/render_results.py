"""Publication-style plots from completed, frozen final exams."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).parent


def main():
    summary=json.loads((ROOT/'summary.json').read_text());selection=json.loads((ROOT/'selection.json').read_text())
    results=summary['final'];selected=selection.get('selected_model',selection['selected_variant'])
    variants=['parent','pool','risk','attention']+(['capacity'] if '129773-capacity' in results else [])+(['focus'] if '129773-focus' in results else [])
    labels={'parent':'Parent figé','pool':'Curriculum seul','risk':'Cinématique','attention':'Attention','capacity':'Capacité','focus':'Focus 4T (+budget)'}
    colors=['#367fa5','#d99539']
    fig,axes=plt.subplots(1,2,figsize=(12.7,4.5),gridspec_kw={'width_ratios':[1.15,1]})
    x=np.arange(len(variants));width=.34
    for index,seed in enumerate((129773,242579)):
        values=[results[f'{seed}-{v}']['controlled']['4T12']*100 for v in variants]
        axes[0].bar(x+(index-.5)*width,values,width,color=colors[index],label=str(seed))
        for a,b in zip(x+(index-.5)*width,values):axes[0].text(a,b+1,f'{b:.1f}',ha='center',fontsize=8)
    axes[0].set_xticks(x,[labels[v] for v in variants],rotation=16,ha='right')
    axes[0].set_title('Quatre tireurs à 12 PV — nouveaux épisodes')
    axes[0].legend(title="Graine d’apprentissage",fontsize=8)
    groups=['grouped','synchronized','geometry'];names=['Ennemis groupés','Tirs synchronisés','Grande salle']
    x=np.arange(3);width=.19
    for index,(seed,variant) in enumerate([(129773,'parent'),(129773,selected),(242579,'parent'),(242579,selected)]):
        values=[results[f'{seed}-{variant}'][suite]['hard_clear_rate']*100 for suite in groups]
        offset=(index-1.5)*width
        axes[1].bar(x+offset,values,width,color=colors[0 if seed==129773 else 1],alpha=.35 if variant=='parent' else 1,
                    label=f'{seed} '+('parent' if variant=='parent' else labels[selected]))
    axes[1].set_xticks(x,names,rotation=12,ha='right')
    axes[1].set_title('Facteurs réservés — 3/4 tireurs à 12 PV')
    axes[1].legend(fontsize=7)
    for ax in axes:
        ax.set_ylim(0,100);ax.set_ylabel('Victoires sans dégât (%)');ax.spines[['top','right']].set_visible(False)
        ax.yaxis.grid(True,alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('IsaacRL synthétique — architectures à exposition égale ; focus au budget supplémentaire ; aucun jeu réel',fontsize=10)
    fig.tight_layout();fig.savefig(ROOT/'crossfire.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.1),sharey=True)
    palette={'pool':'#397f9c','risk':'#b17444','attention':'#7a6fa6','capacity':'#74854a'}
    for ax,seed in zip(axes,(129773,242579)):
        for variant,arm in summary['training'][str(seed)].items():
            folder=ROOT/f'paired-seed{seed}'
            report=json.loads((folder/'report.json').read_text())
            if variant not in report['arms']:report=json.loads((ROOT/f'capacity-seed{seed}/report.json').read_text())
            original=report['arms'][variant]
            def hard(cohorts):
                values=[cell['clear_rate'] for cohort in cohorts for cell in cohort['cells'] if cell['scene']['shooters']==4 and cell['scene']['shooter_hp']==12]
                return np.mean(values)*100
            x=[0]+[r['updates']*32768/1e6 for r in original['checks']]
            y=[hard(original['baseline'])]+[hard(r['cohorts']) for r in original['checks']]
            ax.plot(x,y,marker='.',label=labels[variant],color=palette[variant])
        focus=summary['focus'].get(str(seed))
        if focus:
            variant=selection['selected_variant'];arm=focus['arms'][variant]
            parent=json.loads(Path(focus['config']['parent']).with_suffix('.meta.json').read_text())
            offset=parent['updates']*32768/1e6
            x=[offset]+[offset+r['updates']*32768/1e6 for r in arm['checks']]
            y=[hard(arm['baseline'])]+[hard(r['cohorts']) for r in arm['checks']]
            ax.plot(x,y,'--o',markersize=3,color='#384c3d',label='Focus 4T (+budget)')
        ax.set(title='Graine '+str(seed),xlabel='Transitions supplémentaires (millions)',ylim=(0,100))
        ax.yaxis.grid(True,alpha=.2);ax.spines[['top','right']].set_visible(False);ax.legend(fontsize=7)
    axes[0].set_ylabel('4 tireurs à 12 PV — développement sans dégât (%)')
    fig.suptitle('Progression sur les deux cohortes de développement ; aucun score final utilisé pour choisir les poids',fontsize=10)
    fig.tight_layout();fig.savefig(ROOT/'learning-curves.png',dpi=170);plt.close(fig)


if __name__=='__main__':main()
