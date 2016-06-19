"""
================================================================
Compute sparse inverse solution with mixed norm: MxNE and irMxNE
================================================================

Runs (ir)MxNE (L1/L2 or L0.5/L2 mixed norm) inverse solver.
L0.5/L2 is done with irMxNE which allows for sparser
source estimates with less amplitude bias due to the non-convexity
of the L0.5/L2 mixed norm penalty.

See
Gramfort A., Kowalski M. and Hamalainen, M,
Mixed-norm estimates for the M/EEG inverse problem using accelerated
gradient methods, Physics in Medicine and Biology, 2012
http://dx.doi.org/10.1088/0031-9155/57/7/1937

Strohmeier D., Haueisen J., and Gramfort A.:
Improved MEG/EEG source localization with reweighted mixed-norms,
4th International Workshop on Pattern Recognition in Neuroimaging,
Tuebingen, 2014
DOI: 10.1109/PRNI.2014.6858545
"""
# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

from scipy import linalg

import mne
from mne.datasets import sample
from mne.inverse_sparse import dynamic_map_em
from mne.inverse_sparse.dmapem import transition_matrix
from mne.viz import plot_sparse_source_estimates

print(__doc__)

data_path = sample.data_path()
fwd_fname = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
ave_fname = data_path + '/MEG/sample/sample_audvis-ave.fif'
cov_fname = data_path + '/MEG/sample/sample_audvis-cov.fif'
subjects_dir = data_path + '/subjects'

# Read noise covariance matrix
cov = mne.read_cov(cov_fname)
# Handling average file
condition = 'Left Auditory'
evoked = mne.read_evokeds(ave_fname, condition=condition, baseline=(None, 0))
evoked.crop(tmin=0, tmax=0.3)
# Handling forward solution
fwd = mne.read_forward_solution(fwd_fname, surf_ori=True, force_fixed=True)

ylim = dict(eeg=[-10, 10], grad=[-400, 400], mag=[-600, 600])
evoked.plot(ylim=ylim, proj=True)

###############################################################################
# Run solver
SNR = 5
lam = 1. / SNR**2   # Regularization related to power SNR
b = 3   # Parameter for inv gamma hyper prior, to make it non (little) inform
phi = 0.8   # Temporal autocorrelation of lag 1
maxit = 20
mem_type = 'memmap'

# F_hemis = transition_matrix(fwd['src'], alpha=0.5, dist_weight=False)
# F = linalg.block_diag(F_hemis[0].todense(), F_hemis[1].todense())

F = None

# Compute dmapem inverse solution
stc, nus, cost = dynamic_map_em(fwd, evoked, cov, phi=phi, F=F, lam=lam,
                                nu=None, C=None, b=b, save_nu_iter=True,
                                tol=1e-5, maxit=maxit, mem_type=mem_type,
                                prefix=None, delete_cov=True, verbose=None)


###############################################################################
# View in 2D and 3D ("glass" brain like 3D plot)
plot_sparse_source_estimates(fwd['src'], stc, bgcolor=(1, 1, 1),
                             fig_name="dmapem (cond %s)" % condition,
                             opacity=0.1)
