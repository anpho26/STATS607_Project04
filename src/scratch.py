import numpy as np
from sklearn.mixture import GaussianMixture
from dmm import MixingMeasure, MixingDendrogram

# toy data
rng = np.random.default_rng(0)
X = np.vstack([
    rng.normal(loc=-2, scale=0.5, size=(100, 2)),
    rng.normal(loc=2, scale=0.5, size=(100, 2)),
])

gmm = GaussianMixture(n_components=5, covariance_type="full", random_state=0).fit(X)
mm = MixingMeasure.from_sklearn_gmm(gmm)
dend = MixingDendrogram(mm).build()

print(dend.heights)
print(dend.groups_per_level[-1])     # should be [{0,1,2,3,4}] in some form
print(dend.to_linkage().shape)       # (k0-1, 4)