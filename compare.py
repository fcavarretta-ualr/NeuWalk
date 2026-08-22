import sys
import bootstrap_test as bst
import numpy as np

from neuwalk.analysis.morphologies import load_morphologies

def _padding(vectors, sum_flag=True):
  max_len = max(len(v) for v in vectors)

  if sum_flag:
    tmp = np.zeros(max_len)
    for v in vectors:
      tmp[:len(v)] += v
    return tmp

  ret = []
  for v in vectors:
    tmp = np.zeros(max_len)
    tmp[:len(v)] = v
    ret.append(tmp)
  return ret
  
def get_stats(neurons, label):
  all_bif_cnt = []
  all_tot_len = []
  all_sholl_plots = []
  for fname, nrn in neurons:
<<<<<<< HEAD
    bif_cnt = 0
    tot_len = 0
    sholl_plots = []
=======
    
    bif_cnt = 0
    tot_len = 0
    sholl_plots = []

    assert len(nrn) == 1
>>>>>>> 21a7a56 (last version)
    for ch in nrn[0].children:
      if ch.label == label:
        bif_cnt += ch.bifurcation_count
        tot_len += ch.total_length
        sholl_plots.append(ch.sholl_plot(10))
<<<<<<< HEAD
    sholl_plots = _padding(sholl_plots)
    all_bif_cnt.append(bif_cnt)
    all_tot_len.append(tot_len)
    all_sholl_plots.append(sholl_plots)
  all_sholl_plots = _padding(all_sholl_plots, sum_flag=False)
=======
    sholl_plots = _padding(sholl_plots, sum_flag=True)
    
    all_bif_cnt.append(bif_cnt)
    all_tot_len.append(tot_len)
    all_sholl_plots.append(sholl_plots)
    
  all_sholl_plots = _padding(all_sholl_plots, sum_flag=False)
  
>>>>>>> 21a7a56 (last version)
  return all_bif_cnt, all_tot_len, np.vstack(all_sholl_plots)



        
pop1 = load_morphologies(sys.argv[-2], return_file_names=True)
pop2 = load_morphologies(sys.argv[-1], return_file_names=True)

print('compare basal_dendrite')

pop_stats1 = get_stats(pop1, 'basal_dendrite')
pop_stats2 = get_stats(pop2, 'basal_dendrite')

##mu_len = np.mean(pop_stats2[1])
##sd_len = np.std(pop_stats2[1])
##indices = [ i for i, x in enumerate(pop_stats1[1]) if mu_len - 3*sd_len <= x <= mu_len + 3*sd_len]
##print('indices=', len(indices))
##pop_stats1 = ([pop_stats1[0][i] for i in indices], [pop_stats1[1][i] for i in indices], np.array([pop_stats1[2][i, :] for i in indices]))


<<<<<<< HEAD
print('bif p value for mean', bst.bootstrap_pvalue_mean_diff(pop_stats1[0], pop_stats2[0])[0])
print('bif p value for var', bst.bootstrap_pvalue_var_ratio(pop_stats1[0], pop_stats2[0])[0])

print('len p value for mean', bst.bootstrap_pvalue_mean_diff(pop_stats1[1], pop_stats2[1])[0])
print('len p value for var', bst.bootstrap_pvalue_var_ratio(pop_stats1[1], pop_stats2[1])[0])

for i in range(max(pop_stats1[2].shape[1], pop_stats2[2].shape[1])):
  if i < pop_stats1[2].shape[1]:
    a = pop_stats1[2][:, i]
  else:
    a = np.zeros(pop_stats1[2].shape[0])
    
  if i < pop_stats2[2].shape[1]:
    b = pop_stats2[2][:, i]
  else:
    b = np.zeros(pop_stats2[2].shape[0])

  print(i, bst.bootstrap_pvalue_mean_diff(a, b)[0], bst.bootstrap_pvalue_var_ratio(a, b)[0])
=======
print('bif p value for mean', bst.bootstrap_pvalue_mean_diff(pop_stats1[0], pop_stats2[0])[0], np.mean(pop_stats1[0]), np.mean(pop_stats2[0]))
print('bif p value for var', bst.bootstrap_pvalue_var_ratio(pop_stats1[0], pop_stats2[0])[0], np.std(pop_stats1[0]), np.std(pop_stats2[0]))

print('len p value for mean', bst.bootstrap_pvalue_mean_diff(pop_stats1[1], pop_stats2[1])[0], np.mean(pop_stats1[1]), np.mean(pop_stats2[1]))
print('len p value for var', bst.bootstrap_pvalue_var_ratio(pop_stats1[1], pop_stats2[1])[0], np.std(pop_stats1[1]), np.std(pop_stats2[1]))

##for i in range(max(pop_stats1[2].shape[1], pop_stats2[2].shape[1])):
##  if i < pop_stats1[2].shape[1]:
##    a = pop_stats1[2][:, i]
##  else:
##    a = np.zeros(pop_stats1[2].shape[0])
##    
##  if i < pop_stats2[2].shape[1]:
##    b = pop_stats2[2][:, i]
##  else:
##    b = np.zeros(pop_stats2[2].shape[0])
##
##  print(i, bst.bootstrap_pvalue_mean_diff(a, b)[0], bst.bootstrap_pvalue_var_ratio(a, b)[0], np.mean(a), np.std(a), np.mean(b), np.std(b))
>>>>>>> 21a7a56 (last version)
  
    
print('\ncompare apical_dendrite')

pop_stats1 = get_stats(pop1, 'apical_dendrite')
pop_stats2 = get_stats(pop2, 'apical_dendrite')

##mu_len = np.mean(pop_stats2[1])
##sd_len = np.std(pop_stats2[1])
##indices = [ i for i, x in enumerate(pop_stats1[1]) if mu_len - 3*sd_len <= x <= mu_len + 3*sd_len]
##print('indices=', len(indices))
##pop_stats1 = ([pop_stats1[0][i] for i in indices], [pop_stats1[1][i] for i in indices], np.array([pop_stats1[2][i, :] for i in indices]))


<<<<<<< HEAD
print('bif p value for mean', bst.bootstrap_pvalue_mean_diff(pop_stats1[0], pop_stats2[0])[0])
print('bif p value for var', bst.bootstrap_pvalue_var_ratio(pop_stats1[0], pop_stats2[0])[0])

print('len p value for mean', bst.bootstrap_pvalue_mean_diff(pop_stats1[1], pop_stats2[1])[0])
print('len p value for var', bst.bootstrap_pvalue_var_ratio(pop_stats1[1], pop_stats2[1])[0])

for i in range(max(pop_stats1[2].shape[1], pop_stats2[2].shape[1])):
  if i < pop_stats1[2].shape[1]:
    a = pop_stats1[2][:, i]
  else:
    a = np.zeros(pop_stats1[2].shape[0])
    
  if i < pop_stats2[2].shape[1]:
    b = pop_stats2[2][:, i]
  else:
    b = np.zeros(pop_stats2[2].shape[0])

  print(i, bst.bootstrap_pvalue_mean_diff(a, b)[0], bst.bootstrap_pvalue_var_ratio(a, b)[0])
=======
print('bif p value for mean', bst.bootstrap_pvalue_mean_diff(pop_stats1[0], pop_stats2[0])[0], np.mean(pop_stats1[0]), np.mean(pop_stats2[0]))
print('bif p value for var', bst.bootstrap_pvalue_var_ratio(pop_stats1[0], pop_stats2[0])[0], np.std(pop_stats1[0]), np.std(pop_stats2[0]))

print('len p value for mean', bst.bootstrap_pvalue_mean_diff(pop_stats1[1], pop_stats2[1])[0], np.mean(pop_stats1[1]), np.mean(pop_stats2[1]))
print('len p value for var', bst.bootstrap_pvalue_var_ratio(pop_stats1[1], pop_stats2[1])[0], np.std(pop_stats1[1]), np.std(pop_stats2[1]))

##for i in range(max(pop_stats1[2].shape[1], pop_stats2[2].shape[1])):
##  if i < pop_stats1[2].shape[1]:
##    a = pop_stats1[2][:, i]
##  else:
##    a = np.zeros(pop_stats1[2].shape[0])
##    
##  if i < pop_stats2[2].shape[1]:
##    b = pop_stats2[2][:, i]
##  else:
##    b = np.zeros(pop_stats2[2].shape[0])
##
##  print(i, bst.bootstrap_pvalue_mean_diff(a, b)[0], bst.bootstrap_pvalue_var_ratio(a, b)[0], np.mean(a), np.std(a), np.mean(b), np.std(b))
>>>>>>> 21a7a56 (last version)
