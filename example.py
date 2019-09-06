# Check settings
import tensorflow as tf
from keras import backend as K
print("-------------------------------------------")
print("GPU available: ", tf.test.is_gpu_available())
print("Keras backend: ", K.backend())
print("-------------------------------------------")

import numpy as np

import argparse
parser = argparse.ArgumentParser(description='Test program for distance decorrelation.')
parser.add_argument('--kappa', help="parameter determining decorrelation strength")
args = parser.parse_args()
kappa = float(args.kappa)
print(kappa)

# Load layers from keras
from keras.layers import Dense, Input, Concatenate, Flatten, BatchNormalization, Dropout, LeakyReLU
from keras.models import Sequential, Model
from keras.losses import binary_crossentropy
from Disco_tf import distance_corr 


# build one block for each dense layer
def get_block(L, size):
    L = BatchNormalization()(L)

    L = Dense(size)(L)
    L = Dropout(0.5)(L)
    L = LeakyReLU(0.2)(L)
    return L

# baseline correlation function
def binary_cross_entropy(y_true, y_pred):
    
    return binary_crossentropy(y_true, y_pred)

# define new loss with distance decorrelation
def decorr(var_1, var_2, weights):

    def loss(y_true, y_pred):
        #return binary_crossentropy(y_true, y_pred) + distance_corr(var_1, var_2, weights)
        #return distance_corr(var_1, var_2, weights)
        return binary_crossentropy(y_true, y_pred) + kappa * distance_corr(var_1, var_2, weights)
        #return binary_crossentropy(y_true, y_pred)

    return loss


# load and split dataset
input_dir = "/work/creissel/DATA/binaryclassifier"
allX = { feat : np.load(input_dir+'/%s.npy' % feat) for feat in ["jets","leps","met"] }
X = list(allX.values())
y = np.load(input_dir+'/target.npy')

from sklearn.model_selection import train_test_split
split = train_test_split(*X,y , test_size=0.1, random_state=42)
train = [ split[ix] for ix in range(0,len(split),2) ]
test = [ split[ix] for ix in range(1,len(split),2) ]
X_train, y_train = train[0:3], train[-1]
X_test, y_test = test[0:3], test[-1]

X_train.append(np.ones(len(y_train)))
X_test.append(np.ones(len(y_train)))

# Setup network
# make inputs
jets = Input(shape=X_train[0].shape[1:])
f_jets = Flatten()(jets)
leps = Input(shape=X_train[1].shape[1:])
f_leps = Flatten()(leps)
met = Input(shape=X_train[2].shape[1:])
i = Concatenate(axis=-1)([f_jets, f_leps, met])
sample_weights = Input(shape=(1,))
#setup trainable layers
d1 = get_block(i, 1024)
d2 = get_block(d1, 1024)
d3 = get_block(d2, 512)
d4 = get_block(d3, 256)
d5 = get_block(d4, 128)
o = Dense(1, activation="sigmoid")(d5)

model = Model(inputs=[jets,leps,met, sample_weights], outputs=o)
model.summary()

# Compile model
from keras.optimizers import Adam
opt = Adam(lr=0.001)
model.compile(optimizer=opt, loss=decorr(jets[:,0,0], o[:,0], sample_weights[:,0]))
#model.compile(optimizer=opt, loss="binary_crossentropy")

# Train model
model.fit(x=X_train, y=y_train, epochs=20, batch_size=10000, validation_split=0.1)

# Evaluate model
y_train_predict = model.predict(X_train, batch_size=10000)
y_test_predict = model.predict(X_test, batch_size=10000)
from sklearn.metrics import roc_auc_score
auc_train = roc_auc_score(y_train, y_train_predict)
auc_test = roc_auc_score(y_test, y_test_predict)
print("area under ROC curve (train sample): ", auc_train)
print("area under ROC curve (test sample): ", auc_test)

# plot correlation
x = X_test[0][:,0,0]
y = y_test_predict[:,0]
corr = np.corrcoef(x, y)
print("correlation ", corr[0][1])

np.save("/work/creissel/TTH/sw/CMSSW_9_4_9/src/TTH/DNN/DisCo/results/"+str(kappa)+"__leading_jet_pt", x)
np.save("/work/creissel/TTH/sw/CMSSW_9_4_9/src/TTH/DNN/DisCo/results/"+str(kappa)+"__classifier", y)
np.save("/work/creissel/TTH/sw/CMSSW_9_4_9/src/TTH/DNN/DisCo/results/"+str(kappa)+"__truth", y_test)

#fig = plt.figure()
#plt.scatter(x,y)
#plt.xlabel("leading jet pt")
#plt.ylabel("classifier output")
#fig.savefig("/work/creissel/TTH/sw/CMSSW_9_4_9/src/TTH/DNN/DisCo/corr.png")
