import skimage.color
import skimage.io
import skimage.transform
import skimage.feature

import sklearn
from sklearn.externals import joblib

clf = joblib.load("train_model.m")

im = skimage.io.imread('demo.jpg')
im = skimage.color.rgb2gray(im)

im = skimage.transform.resize(im, (256, 256))
hf = skimage.feature.hog(im, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1))
r = clf.predict_proba([hf])

print (r)