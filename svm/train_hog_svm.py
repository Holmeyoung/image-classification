import os

import skimage.color
import skimage.feature
import skimage.io
import skimage.transform
import sklearn.svm
import argparse
from tqdm import tqdm

def read_and_preprocess(im_path):
    im = skimage.io.imread(im_path)
    im = skimage.color.rgb2gray(im)
    im = skimage.transform.resize(im, (256, 256))
    return im


def get_data_tr(path):
    X = []
    Y = []
    categories = os.listdir(path)
    for c in categories:
        path_c = os.path.join(path, c)
        for entry in tqdm(os.scandir(r'%s' % (path_c))):
            try:
                im = read_and_preprocess(entry.path)
                hf = skimage.feature.hog(im, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1))
            except:
                pass
            else:
                X.append(hf)
                Y.append(c)

    return X, Y


def get_data_te(path):
    X = []
    Y = []
    categories = os.listdir(path)
    for c in categories:
        path_c = os.path.join(path, c)
        for entry in tqdm(os.scandir(r'%s' % (path_c))):
            try:
                im = read_and_preprocess(entry.path)
                hf = skimage.feature.hog(im, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1))
            except:
                pass
            else:   
                X.append(hf)
                Y.append(c)

    return X, Y

if __name__ == "__main__":
    parser  =argparse.ArgumentParser()
    parser.add_argument('--train', type = str, required = True, help = 'train data path')
    parser.add_argument('--test', type = str, required = True, help = 'test data path')
    args = parser.parse_args()

    # 训练
    print ('Load train data...')
    Xtr, Ytr = get_data_tr(args.train)
    print ('Training process...')
    clf = sklearn.svm.SVC(probability=True) 
    clf.fit(Xtr, Ytr)

    # 测试
    print ('Load test data...')
    Xte, Yte = get_data_te(args.test)
    print ('Predicting process...')
    r = clf.predict(Xte)
    s = 0
    for i in range(len(r)):
        if r[i] == Yte[i]:
            s += 1
    print('acc:', s / len(r))

    from sklearn.externals import joblib
    joblib.dump(clf, "train_model.m")
