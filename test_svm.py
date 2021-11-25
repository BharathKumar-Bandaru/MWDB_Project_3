import numpy as np
from sklearn import preprocessing
from sklearn.utils import shuffle
import icecream as ic
from sklearn import svm
from tasks.utilities import *
from tasks.custom_svm import *
from tasks.svm import *

fe = "cm"
dim_red = "svd"
k = 10
train_images_path = "Dataset"
filter = "image_id"
latent_semantics, labels_train, X_data  = retrive_features_task1(train_images_path, fe, dim_red, k, filter)
latent_semantics_100, label_test_100, X_test_100  = retrive_features_task1("100", fe, dim_red, k, filter)
latent_semantics_500, label_test_500, X_test_500 = retrive_features_task1("500", fe, dim_red, k, filter)
latent_semantics_1000, label_test_1000, X_test_1000 = retrive_features_task1("1000", fe, dim_red, k, filter)
latent_semantics_2000, label_test_2000, X_test_2000 = retrive_features_task1("2000", fe, dim_red, k, filter)
latent_semantics_3000, label_test_3000, X_test_3000 = retrive_features_task1("3000", fe, dim_red, k, filter)

mean_img_fe = np.mean(X_data, axis=0)
X_data = X_data - mean_img_fe
X_test_100 = X_test_100 - mean_img_fe
X_test_500 = X_test_500 - mean_img_fe
X_test_1000 = X_test_1000 - mean_img_fe
X_test_2000 = X_test_2000 - mean_img_fe
X_test_3000 = X_test_3000 - mean_img_fe

feature_train = np.matmul(X_data, latent_semantics.transpose())
features_test_100 = np.matmul(X_test_100, latent_semantics.transpose())
features_test_500 = np.matmul(X_test_500, latent_semantics.transpose())
features_test_1000 = np.matmul(X_test_1000, latent_semantics.transpose())
features_test_2000 = np.matmul(X_test_2000, latent_semantics.transpose())
features_test_3000 = np.matmul(X_test_3000, latent_semantics.transpose())

sh_fe_tr, sh_la_tr = feature_train, labels_train-1#shuffle(feature_train, labels_train, random_state=0)
# # sh_fe_tr = preprocessing.normalize(sh_fe_tr)
# mean_img_fe = np.mean(sh_fe_tr, axis=0)
# sh_fe_tr = sh_fe_tr - mean_img_fe
# features_test_100 -= mean_img_fe
# features_test_500 -= mean_img_fe
# features_test_1000 -= mean_img_fe
# features_test_2000 -= mean_img_fe
# features_test_3000 -= mean_img_fe
#
#Computing SVM
# clf = MulticlassSVM(C=1, tol=0.01, max_iter=100, random_state=0, verbose=1)
clf = linearSVM()
clf.train(sh_fe_tr, sh_la_tr)
#
pre = clf.predict(features_test_100)
label_test_100 -= 1
ic.ic(np.sum(label_test_100 == pre), np.unique(pre), np.unique(sh_la_tr))
pre = clf.predict(features_test_500)
label_test_500-=1
ic.ic(np.sum(label_test_500 == pre), np.unique(pre))
pre = clf.predict(features_test_1000)
label_test_1000-=1
ic.ic(np.sum(label_test_1000 == pre), np.unique(pre))
pre = clf.predict(features_test_2000)
label_test_2000-=1
ic.ic(np.sum(label_test_2000 == pre), np.unique(pre))
pre = clf.predict(features_test_3000)
label_test_3000-=1
ic.ic(np.sum(label_test_3000 == pre), np.unique(pre))
# # #
# cls = svm.SVC()
# cls.fit(feature_train, labels_train)