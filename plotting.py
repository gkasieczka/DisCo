import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from sklearn.metrics import roc_curve, roc_auc_score

path = "/work/creissel/TTH/sw/CMSSW_9_4_9/src/TTH/DNN/DisCo/results/"

kappa = ["0.0", "0.1", "0.5", "1.0", "1.5", "2.0"]

fpr = {}
tpr = {}
auc = {}
corr = {}
weights = {}
res = {}

for k in kappa:
    x = np.load(path + k + "__leading_jet_pt.npy")
    y = np.load(path + k + "__classifier.npy")
    y_true = np.load(path + k + "__truth.npy")

    auc[k] = roc_auc_score(y_true, y)
    fpr[k], tpr[k], _ = roc_curve(y_true, y)

    # print correlation
    corr[k] = np.corrcoef(x, y)
    #print("correlation ", corr[0][1])

    counts, bins = np.histogram(x, bins=12, weights=y, range=(0,1500))
    counts2, bins = np.histogram(x, bins=12, range=(0,1500))
    print(counts2)
    weights[k] = np.divide(counts, counts2)
    weights[k] = np.nan_to_num(weights[k])
    res[k] = np.polyfit(bins[:-1], weights[k], 1)

    

fig = plt.figure(figsize=(6,6))
for k in kappa:
    plt.plot(fpr[k], tpr[k], label = k + " (AUC: %.3f)" % auc[k])
plt.legend()
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.xlim([0,1.])
plt.ylim([0,1.])
fig.savefig("ROC.png")

fig = plt.figure()
for k in kappa:
    plt.step(bins[:-1],weights[k], label=k + " (corr: %.2f, slope: %.2f)" % (corr[k][0][1], res[k][1]) )
plt.legend()
plt.xlabel("leading jet pt")
plt.ylabel("classifier output")
fig.savefig("corr.png")
