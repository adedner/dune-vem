from math import log
import pickle, sys, itertools
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('ps',   usedistiller='xpdf')
plt.rc('savefig', format='eps')

colors = itertools.cycle(['blue','red','green','yellow','cyan']) # ,'purple','cyan','olive'])
markers = itertools.cycle(['D','<','>','s','o']) #, 'B','^','v'])

polOrder = 5

with open("errors_p"+str(polOrder)+".dump", 'rb') as f:
    methods,spaceSize,l2errors,h1errors = pickle.load(f)

def plot(errors,ylabel,sub):
    # plt.subplot(sub)
    for i in range(len(methods)):
        c = next(colors)
        name = methods[i][0]
        try:
            if not methods[i][2]["conforming"]:
                name = name + "-CR"
        except KeyError:
            pass
        plt.plot(spaceSize[i::len(methods)],errors[2*i::2*len(methods)],
                marker=next(markers),markersize=12.,markeredgewidth=2.,
                markeredgecolor=c, fillstyle='none',
                linestyle='--', linewidth=4., color=c,
                label=name)
    plt.legend(loc=3)
    plt.grid(b=True, which='major', color='0', linestyle='-',linewidth=2.)
    plt.grid(b=True, which='minor', color='0', linestyle='--')
    plt.xlabel("dofs",fontsize=20)
    plt.ylabel(ylabel,fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=20)

plot(h1errors,"$H^1$-error",121)
plt.xscale('log')
plt.yscale('log')
plt.savefig("h1errors_"+str(polOrder))
plt.clf()
plot(l2errors,"$L^2$-error",122)
plt.xscale('log')
plt.yscale('log')
plt.savefig("l2errors_"+str(polOrder))
plt.clf()


sizes2 = [s for s in spaceSize for _ in range(2)]
l2eocs = [2*log(e1/e0)/log(s0/s1) for e1,e0,s1,s0 in zip(l2errors[2*len(methods)::],l2errors,sizes2[2*len(methods)::],sizes2)]
h1eocs = [2*log(e1/e0)/log(s0/s1) for e1,e0,s1,s0 in zip(h1errors[2*len(methods)::],h1errors,sizes2[2*len(methods)::],sizes2)]

spaceSize = spaceSize[len(methods)::]
plot(h1eocs,"$H^1$-eoc",121)
plt.xscale('log')
plt.yscale('linear')
plt.savefig("h1eocs_"+str(polOrder))
plt.clf()
plot(l2eocs,"$L^2$-eoc",122)
plt.xscale('log')
plt.yscale('linear')
plt.savefig("l2eocs_"+str(polOrder))
plt.clf()
