#!/usr/bin/env python3

_print = print
def print(*args,**kwargs):
    _print(*args,**kwargs,flush=True)

import argparse
import pickle
import gc
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica",
    "font.size": 18,
})

parser = argparse.ArgumentParser(prog="kappaplot")
parser.add_argument("--linear", action=argparse.BooleanOptionalAction, default=False)
args = parser.parse_args()

linear = args.linear
filename = f"kappa_{'linear' if linear else 'hjb'}"
print("Reading from file",filename)

#############################################################

with open(f"{filename}.dump", "rb") as f:
    errors = pickle.load(f)
aSet = set()
sSet = set()
for k in errors.keys():
    print(k)
    proj,o,a = k[0],k[1],k[2][0]
    aSet.add(a)
    for se in errors[k]:
        for e in se:
            sSet.add(e[0])
aSet = sorted(aSet,reverse=True)# [:3]
sSet = sorted(sSet,reverse=True)# [:3]
print(aSet,sSet)

colors = ["r","g","b","c","sienna","b","orange","b","g"]
markers = ['o','s','x','s']
markerFill = ["full","none","top"]
styles = ['--','-.',':','-']


orders = [3,4]
for e,eName in enumerate(["L2","H1","H2"]): # each norm gets own figure
    fig,axs = plt.subplots(len(aSet)+len(orders),
                           len(sSet)+len(orders),
                           sharey=True,
                           figsize=( 5*(len(sSet)+len(orders))+5,
                                     5*(len(aSet)+len(orders)) )
                          )
    figEoc,axsEoc = plt.subplots(len(aSet),len(sSet),
                           figsize=(5*len(sSet)+10,5*len(aSet)))
    for r,a in enumerate(aSet):
        for c,s in enumerate(sSet):
            xlabel = f"s={s}" if r==0 else None
            ylabel = f"a={a}" if c==0 else None
            print(r,c,a,s,xlabel,ylabel)
            ax = axs[r][c]
            axEoc = axsEoc[r][c]
            for proj,projName in enumerate(["original","equal(l)","equal(l-1)"]):
                # ,"eqaul(l-2)"]):
                for oenum,o in enumerate(orders):
                    label=f"{projName}, l={o}" if r==0 and c==0 else None
                    style = {"color":colors[proj],
                             "linestyle":styles[proj],
                             "marker":markers[o-3],
                             "fillstyle":markerFill[o-3],
                             "markersize":10,
                             "linewidth":3
                           }
                    key = (proj-1,o,(a,sSet[-1],linear))
                    if not key in errors:
                        continue

                    err = []
                    for y in errors[key]:
                        for x in y:
                            if x[0] == s:
                                err.append( x[1][e] )
                    err = np.array(err)

                    h = np.array([ 0.5**i for i in range(10) ])
                    N = len(err)
                    ax.loglog(h[:N],err,**style, label=label)
                    eocs = np.log( err[1:] / err[:-1] ) / np.log(0.5)
                    # eocs[eocs<0] = 0 # -np.nan
                    # if o==3 and e==2: print(key,err,eocs)
                    axEoc.semilogx(h[1:N], eocs, **style, label=label)

                    ##########################

                    axStab = axs[r][len(sSet)+oenum]
                    for y in errors[key]:
                        errStab = []
                        for x in y:
                            errStab.append( x[1][e] )
                        axStab.loglog(sSet[:len(errStab)],errStab,**style)

                    ##########################

            ax.grid()
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.xaxis.set_label_position('top')

            fig.legend(loc='outside center right')
            fig.suptitle(f'{eName} error on cube grid', fontsize=32)

            axEoc.grid()
            axEoc.set_ylim([3+1-e-1.5, o+1-e+0.5])
            axEoc.set_xlabel(xlabel)
            axEoc.set_ylabel(ylabel)
            figEoc.legend(loc='outside center right')
            figEoc.suptitle(f'{eName} error on cube grid', fontsize=32)

        for oenum,o in enumerate(orders):
            axStab = axs[r][len(sSet)+oenum]
            axStab.set_facecolor('0.8')
            axStab.grid()
            axStab.invert_xaxis()
            if r==len(aSet)-1:
                axs[r][len(sSet)+oenum].set_xlabel(f"order={o}: s vs. error")

    ########################################

    for c,s in enumerate(sSet):
        for oenum,o in enumerate(orders):
            axA = axs[len(aSet)+oenum][c]
            for proj,projName in enumerate(["original","equal(l)","equal(l-1)"]):
                style = {"color":colors[proj],
                         "linestyle":styles[proj],
                         "marker":markers[o-3],
                         "fillstyle":markerFill[o-3],
                         "markersize":10,
                         "linewidth":3
                       }
                for i in range(10):
                    errA = []
                    for r,a in enumerate(aSet):
                        key = (proj-1,o,(a,sSet[-1],linear))
                        if not key in errors:
                            continue
                        if i<len(errors[key]):
                            for x in errors[key][i]:
                                if x[0] == s:
                                    errA.append( x[1][e] )
                    if len(errA) == 0: break
                    axA.loglog(aSet[:len(errA)],errA,**style)
            axA.set_facecolor('0.8')
            axA.grid()
            axA.invert_xaxis()
            if c==0:
                axA.set_ylabel(f"order={o}: a vs. error")

    fig.savefig(f"{filename}_{eName}.pdf")
    figEoc.savefig(f"{filename}EOC_{eName}.pdf")
