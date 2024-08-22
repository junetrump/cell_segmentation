import direction_model
from ioUtils import *
import math
import lossFunction
import tensorflow as tf
import numpy as np
import sys
import os
import scipy.io as sio
import re
import time
from tqdm import tqdm
import csv

VGG_MEAN = [177.28922308, 190.91891843, 204.30707066]

tf.set_random_seed(0)

def initialize_model(outputChannels, wd=1e-5, modelWeightPaths=None):
    fuseChannels=256
    params = {"direction/conv1_1": {"name": "direction/conv1_1", "shape": [3,3,4,64], "std": None, "act": "relu"},
            "direction/conv1_2": {"name": "direction/conv1_2", "shape": [3,3,64,64], "std": None, "act": "relu"},
            "direction/conv2_1": {"name": "direction/conv2_1", "shape": [3,3,64,128], "std": None, "act": "relu"},
            "direction/conv2_2": {"name": "direction/conv2_2", "shape": [3,3,128,128], "std": None, "act": "relu"},
            "direction/conv3_1": {"name": "direction/conv3_1", "shape": [3,3,128,256], "std": None, "act": "relu"},
            "direction/conv3_2": {"name": "direction/conv3_2", "shape": [3,3,256,256], "std": None, "act": "relu"},
            "direction/conv3_3": {"name": "direction/conv3_3", "shape": [3,3,256,256], "std": None, "act": "relu"},
            "direction/conv3_4": {"name": "direction/conv3_4", "shape": [3,3,256,256], "std": None, "act": "relu"},


            "direction/conv4_1": {"name": "direction/conv4_1", "shape": [3,3,256,512], "std": None, "act": "relu"},
            "direction/conv4_2": {"name": "direction/conv4_2", "shape": [3,3,512,512], "std": None, "act": "relu"},
            "direction/conv4_3": {"name": "direction/conv4_3", "shape": [3,3,512,512], "std": None, "act": "relu"},
            "direction/conv4_4": {"name": "direction/conv4_4", "shape": [3,3,512,512], "std": None, "act": "relu"},

            "direction/conv5_1": {"name": "direction/conv5_1", "shape": [3,3,512,512], "std": None, "act": "relu"},
            "direction/conv5_2": {"name": "direction/conv5_2", "shape": [3,3,512,512], "std": None, "act": "relu"},
            "direction/conv5_3": {"name": "direction/conv5_3", "shape": [3,3,512,512], "std": None, "act": "relu"},
            "direction/conv5_4": {"name": "direction/conv5_4", "shape": [3,3,512,512], "std": None, "act": "relu"},


            "direction/fcn5_1": {"name": "direction/fcn5_1", "shape": [5,5,512,512], "std": None, "act": "relu"},
            "direction/fcn5_2": {"name": "direction/fcn5_2", "shape": [1,1,512,512], "std": None, "act": "relu"},
            "direction/fcn5_3": {"name": "direction/fcn5_3", "shape": [1,1,512,fuseChannels], "std": 1e-2, "act": "relu"},

            "direction/upscore5_3": {"name": "direction/upscore5_3", "ksize": 8, "stride": 4, "outputChannels": fuseChannels},
            "direction/fcn4_1": {"name": "direction/fcn4_1", "shape": [5,5,512,512], "std": None, "act": "relu"},
            "direction/fcn4_2": {"name": "direction/fcn4_2", "shape": [1,1,512,512], "std": None, "act": "relu"},
            "direction/fcn4_3": {"name": "direction/fcn4_3", "shape": [1,1,512,fuseChannels], "std": 1e-3, "act": "relu"},
            "direction/upscore4_3": {"name": "direction/upscore4_3", "ksize": 4, "stride": 2, "outputChannels": fuseChannels},
            "direction/fcn3_1": {"name": "direction/fcn3_1", "shape": [5,5,256,256], "std": None, "act": "relu"},
            "direction/fcn3_2": {"name": "direction/fcn3_2", "shape": [1,1,256,256], "std": None, "act": "relu"},
            "direction/fcn3_3": {"name": "direction/fcn3_3", "shape": [1,1,256,fuseChannels], "std": 1e-4, "act": "relu"},
            "direction/fuse3_1": {"name": "direction/fuse_1", "shape": [1,1,fuseChannels*3, 512], "std": None, "act": "relu"},
            "direction/fuse3_2": {"name": "direction/fuse_2", "shape": [1,1,512,512], "std": None, "act": "relu"},
            "direction/fuse3_3": {"name": "direction/fuse_3", "shape": [1,1,512,outputChannels], "std": None, "act": "lin"},
            "direction/upscore3_1": {"name": "direction/upscore3_1", "ksize": 8, "stride": 4, "outputChannels":outputChannels}}

    return direction_model.Network(params, wd=wd, modelWeightPaths=modelWeightPaths)

def forward_model(model, feeder, outputSavePath):
    with tf.Session() as sess:
        tfBatchImages = tf.placeholder("float", shape=[None, 256, 256, 3])
        tfBatchSS = tf.placeholder("float", shape=[None, 256, 256])
        tfBatchSSMask = tf.placeholder("float", shape=[None, 256, 256])

        with tf.name_scope("model_builder"):
            print("attempting to build model")
            model.build(tfBatchImages, tfBatchSS, tfBatchSSMask)
            print("built the model")
        sys.stdout.flush()

        init = tf.initialize_all_variables()
        sess.run(init)

        for i in range(int(math.floor(feeder.total_samples() / batchSize))):
            imageBatch, ssBatch, ssMaskBatch, idBatch = feeder.next_batch()

            outputBatch = sess.run(model.output, feed_dict={tfBatchImages: imageBatch, tfBatchSS: ssBatch, tfBatchSSMask: ssMaskBatch})

            for j in range(len(idBatch)):
                outputFilePath = os.path.join(outputSavePath, idBatch[j]+'.mat')
                outputFileDir = os.path.dirname(outputFilePath)

                if not os.path.exists(outputFileDir):
                    os.makedirs(outputFileDir)

                sio.savemat(outputFilePath, {"dir_map": outputBatch[j]}, do_compression=True)

                print("processed image %d out of %d"%(j+batchSize*i, feeder.total_samples()))
def save_loss_to_csv(loss_values, csv_file_path):
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(loss_values)

def train_model(model, outputChannels, learningRate, trainFeeder, valFeeder, modelSavePath=None, savePrefix=None, initialIteration=1):

    csv_file_path = os.path.join(modelSavePath, 'loss_values.csv')
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["iteration", "Validation Loss", "Angle MSE", "Exceed 45", "Exceed 22.5"])

    with tf.Session() as sess:
        tfBatchImages = tf.placeholder("float", shape=[None, 256, 256, 3])
        tfBatchGT = tf.placeholder("float", shape=[None, 256, 256, 2])
        tfBatchWeight = tf.placeholder("float", shape=[None, 256, 256])
        tfBatchSS = tf.placeholder("float", shape=[None, 256, 256])
        tfBatchSSMask = tf.placeholder("float", shape=[None, 256, 256])

        with tf.name_scope("model_builder"):
            print("attempting to build model")
            model.build(tfBatchImages, tfBatchSS, tfBatchSSMask)
            print("built the model")

        sys.stdout.flush()
        loss = lossFunction.angularErrorLoss(pred=model.output, gt=tfBatchGT, weight=tfBatchWeight, ss=tfBatchSS, outputChannels=outputChannels)

        angleError = lossFunction.angularErrorTotal(pred=model.output, gt=tfBatchGT, weight=tfBatchWeight, ss=tfBatchSS, outputChannels=outputChannels)
        numPredicted = lossFunction.countTotal(ss=tfBatchSS)
        numPredictedWeighted = lossFunction.countTotalWeighted(ss=tfBatchSS, weight=tfBatchWeight)
        exceed45 = lossFunction.exceedingAngleThreshold(pred=model.output, gt=tfBatchGT,
                                                        ss=tfBatchSS, threshold=45.0, outputChannels=outputChannels)
        exceed225 = lossFunction.exceedingAngleThreshold(pred=model.output, gt=tfBatchGT,
                                                        ss=tfBatchSS, threshold=22.5, outputChannels=outputChannels)



        train_op = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(loss=loss)

        init = tf.initialize_all_variables()

        sess.run(init)

        iteration = initialIteration

        while iteration < 101:
            batchLosses = []
            totalAngleError = 0
            totalExceed45 = 0
            totalExceed225 = 0
            totalPredicted = 0
            totalPredictedWeighted = 0

            for k in tqdm(range(int(math.floor(valFeeder.total_samples() / batchSize)))):
                imageBatch, gtBatch, weightBatch, ssBatch, ssMaskBatch, _ = valFeeder.next_batch()

                batchLoss, batchAngleError, batchPredicted, batchPredictedWeighted, batchExceed45, batchExceed225 = sess.run(
                    [loss, angleError, numPredicted, numPredictedWeighted, exceed45, exceed225],
                    feed_dict={tfBatchImages: imageBatch,
                               tfBatchGT: gtBatch,
                               tfBatchWeight: weightBatch,
                               tfBatchSS: ssBatch,
                               tfBatchSSMask: ssMaskBatch})
                # print "ran iteration"
                batchLosses.append(batchLoss)
                totalAngleError += batchAngleError
                totalPredicted += batchPredicted
                totalPredictedWeighted += batchPredictedWeighted
                totalExceed45 += batchExceed45
                totalExceed225 += batchExceed225

            if np.isnan(np.mean(batchLosses)):
                print("LOSS RETURNED NaN")
                sys.stdout.flush()
                return 1
            print("%s Itr: %d - val loss: %.3f, angle MSE: %.3f, exceed45: %.3f, exceed22.5: %.3f" % (
                time.strftime("%H:%M:%S"), iteration,
                float(np.mean(batchLosses)), totalAngleError / totalPredictedWeighted,
                totalExceed45 / totalPredicted, totalExceed225 / totalPredicted))
            sys.stdout.flush()

            if (iteration > 0 and iteration % 1 == 0) or checkSaveFlag(modelSavePath):
                modelSaver(sess, modelSavePath, savePrefix, iteration)
            val_loss_values = [
                iteration,
                float(np.mean(batchLosses)),
                totalAngleError / totalPredictedWeighted,
                totalExceed45 / totalPredicted,
                totalExceed225 / totalPredicted
            ]
            save_loss_to_csv(val_loss_values, csv_file_path)

            for j in tqdm(range(int(math.floor(trainFeeder.total_samples() / batchSize)))):
                # print "running batch %d"%(j)
                # sys.stdout.flush()
                imageBatch, gtBatch, weightBatch, ssBatch, ssMaskBatch, _ = trainFeeder.next_batch()
                sess.run(train_op, feed_dict={tfBatchImages: imageBatch,
                                              tfBatchGT: gtBatch,
                                              tfBatchWeight: weightBatch,
                                              tfBatchSS: ssBatch,
                                              tfBatchSSMask: ssMaskBatch})
            iteration += 1

def modelSaver(sess, modelSavePath, savePrefix, iteration, maxToKeep=5):
    allWeights = {}
    for name in [n.name for n in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]:
        param = sess.run(name)
        nameParts = re.split('[:/]', name)
        saveName = nameParts[-4]+'/'+nameParts[-3]+'/'+nameParts[-2]
        allWeights[saveName] = param

    weightsFileName = os.path.join(modelSavePath, savePrefix+'_%03d.mat'%iteration)

    sio.savemat(weightsFileName, allWeights)


def checkSaveFlag(modelSavePath):
    flagPath = os.path.join(modelSavePath, 'saveme.flag')

    if os.path.exists(flagPath):
        return True
    else:
        return False


if __name__ == "__main__":
    outputChannels = 2
    classType = 'unified_CR'
    indices = [1]
    #1=cell
    # 0=car, 1=person, 2=rider, 3=motorcycle, 4=bicycle, 5=truck, 6=bus, 7=train
    savePrefix = "direction_"

    # train = False
    train = True

    if train:
        batchSize = 6
        learningRate = 1e-4
        # learningRateActul = 1e-7
        wd = 1e-5

        # modelWeightPaths = ["/home/zoey/Desktop/files/liub/dwt/DN/new_weight_file_3.mat"]
        initialIteration = 1

        model = initialize_model(outputChannels=outputChannels, wd=wd, modelWeightPaths=None)

        trainFeeder = Batch_Feeder(dataset="cell", indices=indices, train=train, batchSize=batchSize,
                                   padWidth=None, padHeight=None, flip=True, keepEmpty=False)
        # py38dateset
        # trainFeeder.set_paths(idList=read_ids('/home/zoey/Desktop/files/liub/dwt/py38_dataset/data/train/trainlist.txt'),
        #                  imageDir="/home/zoey/Desktop/files/liub/dwt/dataset/py38dataset/train/images_gaosi9",
        #                  gtDir="/home/zoey/Desktop/files/liub/dwt/py38_dataset/data/train/GT",
        #                  ssDir="/home/zoey/Desktop/files/liub/dwt/py38_dataset/data/train/psp")
        # py27dateset
        trainFeeder.set_paths(idList=read_ids('/home/zoey/Desktop/files/liub/dwt/dataset/py27dateset/train/trainlist.txt'),
                         imageDir="/home/zoey/Desktop/files/liub/dwt/dataset/py27dateset/train/images",
                         gtDir="/home/zoey/Desktop/files/liub/dwt/dataset/py27dateset/train/GT",
                         ssDir="/home/zoey/Desktop/files/liub/dwt/dataset/py27dateset/train/psp")


        valFeeder = Batch_Feeder(dataset="cell", indices=indices, train=train, batchSize=batchSize,
                                 padWidth=None, padHeight=None)
        #py27dateset
        valFeeder.set_paths(idList=read_ids('/home/zoey/Desktop/files/liub/dwt/dataset/py27dateset/val/vallist.txt'),
                            imageDir="/home/zoey/Desktop/files/liub/dwt/dataset/py27dateset/val/images",
                         gtDir="/home/zoey/Desktop/files/liub/dwt/dataset/py27dateset/val/GT",
                         ssDir="/home/zoey/Desktop/files/liub/dwt/dataset/py27dateset/val/psp")

        #py38dateset
        # valFeeder.set_paths(idList=read_ids('/home/zoey/Desktop/files/liub/dwt/py38_dataset/data/val/vallist.txt'),
        #                  imageDir="/home/zoey/Desktop/files/liub/dwt/dataset/py38dataset/val/images_gaosi9",
        #                  gtDir="/home/zoey/Desktop/files/liub/dwt/py38_dataset/data/val/GT",
        #                  ssDir="/home/zoey/Desktop/files/liub/dwt/py38_dataset/data/val/psp")


        train_model(model=model, outputChannels=outputChannels,
                    learningRate=learningRate,
                    trainFeeder=trainFeeder, valFeeder=valFeeder,
                    modelSavePath="/home/zoey/Desktop/files/liub/dwt/dataset/py27dateset/models/dn6",savePrefix=savePrefix,
                    # modelSavePath="/home/zoey/Desktop/files/liub/dwt/dataset/py38dataset/zmodels/dn3", savePrefix=savePrefix,
                    initialIteration=initialIteration)
    else:
        batchSize = 4
        modelWeightPaths = ["/home/zoey/Desktop/files/liub/dwt/dataset/py27dateset/models/dn5/direction__028.mat"]

        model = initialize_model(outputChannels=outputChannels, wd=0, modelWeightPaths=modelWeightPaths)

        feeder = Batch_Feeder(dataset="cell", indices=indices, train=train, batchSize=batchSize, padWidth=None, padHeight=None)

        # py38dateset
        # feeder.set_paths(idList=read_ids("/home/zoey/Desktop/files/liub/dwt/dataset/py38dataset/test/testlist.txt"),
        #                  imageDir="/home/zoey/Desktop/files/liub/dwt/dataset/py38dataset/test/gaosi9",
        #                  ssDir="/home/zoey/Desktop/files/liub/dwt/dataset/py38dataset/test/9_sem")

        # py27dateset
        feeder.set_paths(idList=read_ids('/home/zoey/Desktop/files/liub/dwt/dataset/py27dateset/test/testlist.txt'),
                            imageDir="/home/zoey/Desktop/files/liub/dwt/dataset/py27dateset/test/images",
                            ssDir="/home/zoey/Desktop/files/liub/dwt/dataset/py27dateset/test/psp")

        forward_model(model, feeder=feeder,
                      outputSavePath="/home/zoey/Desktop/files/liub/dwt/dataset/py27dateset/test_save/dn3")
