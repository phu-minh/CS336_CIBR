# import packages
from __future__ import print_function
from os.path import dirname, join, realpath, basename, exists
from image_search_pipeline.descriptors import DetectAndDescribe
from image_search_pipeline.information_retrieval import BagOfVisualWords
from image_search_pipeline.information_retrieval import Searcher
from image_search_pipeline.information_retrieval import dist
from scipy.spatial import distance
from redis import Redis
from imutils.feature import FeatureDetector_create, DescriptorExtractor_create
import numpy as np
import progressbar
import argparse
import pickle
import imutils
import json
import cv2
import os
import glob
from PathConfig import *

# construct the argument parser and parse the argument
# ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--dataset", required = True, help = "Path to the directory of indexed images")
# ap.add_argument("-f", "--features_db", required = True, help = "Path to the feature database")
# ap.add_argument("-b", "--bovw_db", required = True, help = "Path to the bag-of-visual-words database")
# ap.add_argument("-c", "--codebook", required = True, help = "Path to the codebook")
# ap.add_argument("-i", "--idf", type = str, help = "Path to inverted document frequencies array")
# ap.add_argument("-r", "--relevant", required = True, help = "Path to the relevant dictionary")
# args = vars(ap.parse_args())

args = {
          'dataset': FilePaths.dataset, 
          'features_db': FilePaths.features_db, 
          'bovw_db': FilePaths.bovw_db, 
          'codebook': FilePaths.codebook, 
          'idf': FilePaths.idf,
          'relevant': []
        }

# initialize the keypoint detector, local invariant descriptor, descriptor pipeline
# distance metric, and inverted document frequency array
detector = FeatureDetector_create("BRISK")
descriptor = DescriptorExtractor_create("RootSIFT")
dad = DetectAndDescribe(detector, descriptor)
distanceMetric = dist.chi2_distance
idf = None

# if the path to the inverted document frequency array was supplied, then load the
# idf array and update
if args["idf"] is not None:
    idf = pickle.loads(open(args["idf"], "rb").read())
    distanceMetric = distance.cosine

# load the codebook vocabulary and initialize the bag-of-visual-words transformer
vocab = pickle.loads(open(args["codebook"], "rb").read())
bovw = BagOfVisualWords(vocab)

# connect to redis and initialize the searcher
redisDB = Redis(host = "localhost", port = 6379, db = 0)
searcher = Searcher(redisDB, args["bovw_db"], args["features_db"], idf = idf, distanceMetric = distanceMetric)

# load the relevant queries dictionary
# relevant = json.loads(open(args["relevant"]).read())
# queryIDs = relevant.keys()

# initialize the accuracies list and timing list
# accuracies = []
# timings = []

# initialize the progress bar
# widgets = ["Evaluating: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
# pbar = progressbar.ProgressBar(maxval = len(queryIDs), widgets = widgets).start()

GT_folder = FilePaths.groundtruth
ap = []
# loop over query-image
for query_file in glob.glob(join(GT_folder, '*_query.txt')):
    query = open(query_file, 'r')
    line_query = query.read()
    query_image, x, y, h, w = line_query.split()[0] + '.jpg', int(float(line_query.split()[1])), int(float(line_query.split()[2])), \
        int(float(line_query.split()[3])), int(float(line_query.split()[4]))

    # print(join(FilePaths.train_set, join(query_image.split('_')[1] ,query_image)))
    # print(exists(join(FilePaths.train_set, join(query_image.split('_')[1] ,query_image))))

    # img = cv2.imread(join(FilePaths.train_set, join(query_image.split('_')[1] ,query_image)))
    # cv2.imshow("Query", imutils.resize(img, width = 320))
    # print(x, y, h, w)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # cv2.waitKey(1)
    # cv2.rectangle(img, pt1=(x,y), pt2=(h,w), color=(255,0,0), thickness=10)
    # cv2.imshow("Query", imutils.resize(img[y:w, x:h], width = 320))
    # cv2.waitKey(0)

    queryImage = cv2.imread(join(FilePaths.train_set, join(query_image.split('_')[1] ,query_image)))[y:w, x:h]
    queryImage = imutils.resize(queryImage, width = 320)
    queryImage = cv2.cvtColor(queryImage, cv2.COLOR_BGR2GRAY)

    # extract features from the query image and construct
    (_, descs) = dad.describe(queryImage)
    hist = bovw.describe(descs).tocoo()

    ret_file_name = 'result_{}.txt'.format(query_image[:-4])
    if not exists(join(FilePaths.result, ret_file_name)):

        # perform search and compute the total number of relevant images in the top-20 results
        sr = searcher.search(hist, numResults = 100)

        result = open(join(FilePaths.result, ret_file_name), 'w')
        for line in set([r[1] for r in sr.results]):
            result.write(line.decode('utf-8').replace(FilePaths.dataset+'\\', '').replace('.jpg', '')+'\n')
        result.close()
    print(ret_file_name, basename(query_file))
    
    import subprocess
    process = subprocess.Popen([FilePaths.compute_ap_exe, query_file.replace('_query.txt', ''), join(FilePaths.result, ret_file_name)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = process.communicate()
    # print(out)
    ap.append(float(out.decode('utf-8')[:-1]))
    # os.system(FilePaths.compute_ap_exe + ' {} {}'.format(query_file.replace('_query.txt', ''), join(FilePaths.result, ret_file_name)))

    # update the evaluation lists
    # accuracies.append(len(inter))
    # timings.append(sr.search_time)
    # pbar.update(i)

# release any pointers allocated by the searcher
searcher.finish()
# pbar.finish()

# show evaluation information to the user
# accuracies = np.array(accuracies)
# timings = np.array(timings)
# print("[INFO] ACCURACY: u = {:.1f}, o = {:.1f}".format(accuracies.mean(), accuracies.std()))
# print("[INFO] TIMINGS: u = {:.1f}, o = {:.1f}".format(timings.mean(), timings.std()))
# print(ap)
print(np.mean(ap))