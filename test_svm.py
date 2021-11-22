from sklearn import preprocessing
from sklearn.utils import shuffle
import icecream as ic

from tasks.utilities import *
from tasks.custom_svm import *

fe = "cm"
dim_red = "svd"
k = 10
train_images_path = "Dataset"

feature_train, labels_train = retrive_features_task1(train_images_path, fe, dim_red, k)
features_test_100, label_test_100 = retrive_features_task1("100", fe, dim_red, k)
features_test_500, label_test_500 = retrive_features_task1("500", fe, dim_red, k)
features_test_1000, label_test_1000 = retrive_features_task1("1000", fe, dim_red, k)
features_test_2000, label_test_2000 = retrive_features_task1("2000", fe, dim_red, k)
features_test_3000, label_test_3000 = retrive_features_task1("3000", fe, dim_red, k)

sh_fe_tr, sh_la_tr = shuffle(feature_train, labels_train, random_state=0)
sh_fe_tr = preprocessing.normalize(sh_fe_tr)

#Computing SVM
clf = MulticlassSVM(C=1, tol=0.01, max_iter=100, random_state=0, verbose=1)
clf.fit(sh_fe_tr, sh_la_tr)

pre = clf.predict(features_test_100)
ic.ic(label_test_100, pre, np.sum(label_test_100 == pre), np.unique(pre))
pre = clf.predict(features_test_500)
ic.ic(label_test_500, pre, np.sum(label_test_500 == pre), np.unique(pre))
pre = clf.predict(features_test_1000)
ic.ic(label_test_1000, pre, np.sum(label_test_1000 == pre), np.unique(pre))
pre = clf.predict(features_test_2000)
ic.ic(label_test_2000, pre, np.sum(label_test_2000 == pre), np.unique(pre))
pre = clf.predict(features_test_3000)
ic.ic(label_test_3000, pre, np.sum(label_test_3000 == pre), np.unique(pre))