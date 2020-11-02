
import numpy as np
import scipy
import scipy.stats
import math 
from scipy import optimize
import MDAnalysis as mda 
import MDAnalysis.analysis.rdf 
import matplotlib.pyplot as plt
import numpy as np
import math 
from itertools import combinations
from itertools import combinations_with_replacement
import MDAnalysis.analysis.distances 
import scipy.cluster
from scipy.cluster.hierarchy import single, fcluster 
import MDAnalysis.core.topologyattrs
import MDAnalysis.analysis.rdf 
from scipy import integrate


# load the trajectories
all_u = []
for r in range(1,6):
    all_u.append(mda.Universe('rep%i/tric.gro'%r,'rep%i/tric.xtc'%r))
    print('loaded trajectory %i of 5'%r)

# define the regions

classes = ['monomer','small','medium1','medium2','medium3','large1']
maxes = [10, 28, 44, 53, 67, 76]
mins = [0, 9, 27, 43, 52, 66]

# put rij counts for every cluster size into distance bins 

bins = 10000

all_rijs = np.zeros([150,bins])
# load what we need 
for r in range(0,5):         
    u = all_u[r]
  
    frame = 1680 # we will look at the last 'frame' frames
    bins = 10000 # how many bins to use in the histogram approximation 
    
    ####################################################################
    
    # cluster things again 
    #
    box = u.dimensions
    cutoff = 8.5 # angstroms, based on second peak of rdf graph 
    cS = u.select_atoms('name C312') # select the atoms
    r1 = [] # get positions 
    for ts in u.trajectory:
        r1.append((cS.positions))
        #print('loading positions frame %i of %i'%(u.trajectory.frame+1, len(u.trajectory)))
    r1 = np.array(r1)
    dist = [] # get all pair distances formatted as flat upper triangles
    for i in range(len(r1)):
        dist.append(MDAnalysis.analysis.distances.self_distance_array(r1[i], box=box))
        #print('loading distances frame %i of %i'%(i+1, len(u.trajectory)))
    dist = np.array(dist)
    #print('clustering.')
    z = [] # perform hierarchical single-linkage clustering 
    for i in range(len(dist)):
        z.append(single(dist[i]))
    z = np.array(z)
    #print('clustering..')
    hierarchy = [] # get clusters using cutoff (in angstroms)
    for i in range(len(z)):
        hierarchy.append(fcluster(z[i], cutoff, criterion='distance'))
    hierarchy = np.array(hierarchy) 
    #
    # select the indices of the atoms in each cluster 
    #        
    DPC = u.select_atoms('resname FOS12')
    #
    clusters = []
    for j in range(-frame,-1):
        clusters1 = []
        for i in range(1, np.amax(hierarchy[j])+1):
            c_inds = np.where(hierarchy[j] == i)
            c_sel = " ".join(list((c_inds[0]+1).astype('str')))
            sel_atoms = DPC.select_atoms('resid %s'%c_sel) # select beads in the cluster
            clusters1.append(sel_atoms)
        clusters.append(np.array(clusters1))
    #
    # clusters == a list with an array for each frame of atom groups for each cluster 
    # find the distances rij for each cluster
    #       
    for l in range(len(clusters)):
        u.trajectory[len(u.trajectory)-frame+l]
    #
        box = u.dimensions
    #            
        # we need an array of all atom positions for each cluster size 
        atom_pos = []
        for i in range(1,151):  
            pos1 = []
            for j in range(len(clusters[l])):
                if len(clusters[l][j])/61 == i:
                    atom = clusters[l][j]
                    pos1.append(atom.positions)
            atom_pos.append(np.array(pos1))
        
        for i in range(len(atom_pos)):
            if len(atom_pos[i]) > 0:
                for j in range(len(atom_pos[i])):
                    heights = np.histogram(mda.analysis.distances.self_distance_array(atom_pos[i][j],box=box),bins=bins,range=(0.5,150.5))[0]
                    all_rijs[i] = all_rijs[i] + heights
    
    print('finished rep %s of 5'%(r+1))

np.save('Analysis/all_clust_rijs.npy',all_rijs)
print('saved all rijs!')

##################################

t_rij = np.load('Analysis/all_clust_rijs.npy',allow_pickle=True)

classes = ['monomer','small','medium1','medium2','medium3','large1']
maxes = [9, 27, 43, 52, 66, 75]
mins = [0, 9, 27, 43, 52, 66]

c_rij = np.zeros([len(classes),len(t_rij[0])])
for i in range(len(classes)):
    c_rij[i] = np.sum(t_rij[mins[i]:maxes[i]],0)

# organize the distances into an array separated by classes

tot_dist = np.sum(t_rij,0)
all_dists = np.zeros([len(classes)+1,len(tot_dist)])
for i in range(len(all_dists)):
    for j in range(len(all_dists[i])):
        if i < len(classes):
            all_dists[i][j] = c_rij[i][j]
        if i == len(classes):
            all_dists[i][j] = tot_dist[j]


x = np.linspace(0.5,150.5,10000) # the x-axis
dx = x[1]-x[0] 

# normalize 

for i in range(len(all_dists)):
    all_dists[i] = all_dists[i]/np.sum(all_dists[i])

# calculate the radius of gyration from the second moment of the pair distances 
# for each class of clusters. 

##################################################################
#                                                                #
#  Rg**2 = trapz(p(r)r**2)/(2*trapz(pr))                         #
#                                                                #
##################################################################

p_r = []
for i in range(len(all_dists)):
    pr1 = []
    for j in range(len(all_dists[i])):
        pr1.append(4*math.pi*all_dists[i][j])
    p_r.append(np.array(pr1))

# second moment of P(r)

sm_pr = []
for i in range(len(all_dists)):
    sm1 = []
    for j in range(len(all_dists[i])):
        sm1.append((x[j]**2)*4*math.pi*all_dists[i][j])
    sm_pr.append(sm1)

Rg = []
for i in range(len(sm_pr)):
    sm = integrate.trapz(sm_pr[i],x=x,dx=dx)
    norm = integrate.trapz(p_r[i],x=x,dx=dx)
    Rg.append((sm/(2*norm))**.5)


#####################################################################################

# plot the pair distance functions for each class of clusters and report the averge radius of gyration

plt.figure(figsize=[5,3])

plt.plot(x[:4000],sm_pr[0][:4000],'r',     label='%i-%i:     %s'%(mins[0]+1,maxes[0],np.round(Rg[0],2)),linewidth=2)
plt.plot(x[:4000],sm_pr[1][:4000],'orange',label='%i-%i: %s'%(mins[1]+1,maxes[1],np.round(Rg[1],2)),linewidth=2)
plt.plot(x[:4000],sm_pr[2][:4000],'gold',  label='%i-%i: %s'%(mins[2]+1,maxes[2],np.round(Rg[2],2)),linewidth=2)
plt.plot(x[:4000],sm_pr[3][:4000],'g',     label='%i-%i: %s'%(mins[3]+1,maxes[3],np.round(Rg[3],2)),linewidth=2)
plt.plot(x[:4000],sm_pr[4][:4000],'b',     label='%i-%i: %s'%(mins[4]+1,maxes[4],np.round(Rg[4],2)),linewidth=2)
plt.plot(x[:4000],sm_pr[5][:4000],'purple',label='%i-%i: %s'%(mins[5]+1,maxes[5],np.round(Rg[5],2)),linewidth=2)
plt.plot(x[:4000],sm_pr[6][:4000],'k',     label='total:   %s'%np.round(Rg[6],2),linewidth=2,linestyle='dashed')

plt.legend(loc='upper right',frameon=False,fontsize=9)
plt.xlabel(r'Distances, $r$ $(\AA)$',fontsize=15)
plt.ylabel(r'4$\pi r^{2}$P($r$)',fontsize=15)

plt.xticks(ticks=[10,20,30,40,50],fontsize=12)
plt.yticks(ticks=[2,4,6,8,10],fontsize=12)
plt.minorticks_on()
plt.ylim(0,7)
plt.xlim(0,60)
plt.tight_layout()
plt.savefig('figures/pr_second_moment_with_tot.pdf')
plt.show()




        
            
            