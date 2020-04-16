# dtw for seoul narrative data / modified for VAE+GAN+AC code
# Using preprocessed npz original file.
# Trying multi processing and can scan sub dir

# Directory structure (original source files == test dataset)
# -source_folder
# --testsetNor #sourceDir
# ---[spkId]
# ----[spk_id_sn]_[fileName].wav (fileName: t[num]_s[num])

# Directory structure (converted files)
# -testFolder
# --testName
# ---epoch
# ---[spkId_src]_to_[spkId_trg]
# ----[spkId_src]_to_[spkId_trg]_[[fileName]].wav

import numpy as np
from scipy.spatial.distance import euclidean
import librosa
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from multiprocessing import Pool
import time
import math

from dtw import dtw
from utils_world import *

speakers = ['F1','M1','F2','M2']
test_lists = ['1']
#CCGAN_sn_01lr_10.ckpt
testDirs = []
targetEpoch=1800
for each in test_lists:
    curTest = 'CCGAN_sn_4spk_'+each+'_'+str(targetEpoch)+'.ckpt'
    testDirs.append(curTest)

print('tests:')
for each in testDirs:
    print(' {}'.format(each))

sourceDir = '../seoulNarrative_4spk_v3/testSet'
sourceWavs = {}
sourceWavs_path = {} #just for checking

mfcc_dim = 36 #36

outFolder = './logs_MSD/'
resultFileName_base = outFolder+'MSD_result_'
if not os.path.exists(outFolder):
    os.makedirs(outFolder)

counter = 0
for eachSpk in speakers:
    for eachFile in os.listdir(os.path.join(sourceDir, eachSpk)):
        if ".wav" in eachFile:
            fileNameParts = eachFile.split('.')[0].split('_')#fv01_t09_s09
            fileName_pure = fileNameParts[1]+'_'+fileNameParts[2]
            curKey = eachSpk + '_' + fileName_pure
            curFile_path = os.path.join(sourceDir, eachSpk, eachFile)
            curData, sr = librosa.load(curFile_path, sr = 16000, mono = True)
            _f0, _timeaxis, _sp, _ap, mfcc = world_decompose(wav = curData, fs = sr, frame_period = 5.0, num_mcep=mfcc_dim)
            mfcc = np.fft.fft(mfcc.T,50)
            sourceWavs[curKey] = mfcc
            sourceWavs_path[curKey] = curFile_path
            #print(cur_wav_file_loc)
            counter+=1
    print('{} preprosess done.'.format(eachSpk))

print('total source file: {}'.format(counter))

dist = lambda x,y: np.linalg.norm(x-y)
#dist = lambda x,y: np.linalg.norm((x-y)**2)

def calculateMCD_manual_mfcc(target_converted_file_path, compare_original_file_key, target_converted_source_id, compare_original_target_id):
    pure_file_name = compare_original_file_key[3:]
    #print("Calculating: <file {}> / <convType {}> / < {} -> {} >".format(pure_file_name, convType, target_converted_source_id, compare_original_target_id))
    #DTW
    target_conv_wav_data, sr = librosa.load(target_converted_file_path, sr = 16000, mono = True)
    #target_conv_wav_mfcc = librosa.feature.mfcc(y=target_conv_wav_data, sr=sr, n_mfcc=mfcc_dim).T
    #compare_ori_wav_mfcc = librosa.feature.mfcc(y=source_wavs[compare_original_file_key], sr=sr, n_mfcc=mfcc_dim).T
    #distance, path = fastdtw(compare_ori_wav_mfcc, target_conv_wav_mfcc, dist=euclidean)

    #MCD
    _f0, _timeaxis, _sp, _ap, target_conv_wav_mfcc = world_decompose(wav = target_conv_wav_data, fs = sr, frame_period = 5.0, num_mcep=mfcc_dim)
    #print('trg: {}'.format(target_conv_wav_mfcc.shape))
    #_, _, sp_com_ori, _ = world_decompose(wav = source_wavs[compare_original_file_key], fs = sr, frame_period = 5.0)
    try:#F*** Error!
        compare_ori_wav_mfcc = sourceWavs[compare_original_file_key]
        #print('ori: {}'.format(compare_ori_wav_mfcc.shape))
    except:
        print('error: <{}>'.format(target_converted_file_path))
        print('error key: <{}>'.format(compare_original_file_key))
    target_conv_wav_mfcc = np.fft.fft(target_conv_wav_mfcc.T,50)

    MSD = np.mean(np.abs(np.sqrt(np.power((target_conv_wav_mfcc-compare_ori_wav_mfcc),2))))

    resultLine = "file: " + pure_file_name + ' ' + target_converted_source_id + " -> " + compare_original_target_id + " MSD: " + str(MSD)+"\n"
    return resultLine

def calculateMCD_mfcc(workData):
    #[target_converted_file_path, compare_original_file_key, model_epoch, target_converted_source_id, compare_original_target_id]
    target_converted_file_path = workData[0]
    compare_original_file_key = workData[1]
    target_converted_source_id = workData[2]
    compare_original_target_id = workData[3]
    return calculateMCD_manual_mfcc(target_converted_file_path, compare_original_file_key, target_converted_source_id, compare_original_target_id)

def Make_workList(convDir):
    workList = [] #[target_converted_file_path, compare_original_file_key, model_epoch, target_converted_source_id, compare_original_target_id]
    for eachDir in os.listdir(convDir): #
        epoch = eachDir.split('_')[-1]
        for eachWay in os.listdir(os.path.join(convDir, eachDir)):
            curSrc, _, curTrg = eachWay.split('_')
            for eachFile in os.listdir(os.path.join(convDir, eachDir, eachWay)):#ex) F1_to_F2_fv01_t09_s09.wav
                FileNameParts = eachFile.split('.')[0].split('_')
                pureFileName =FileNameParts[-2] + '_' + FileNameParts[-1]
                curFile_path = os.path.join(convDir, eachDir, eachWay, eachFile)
                curKey = curTrg + '_' + pureFileName # [target]_[file_name] compare original file key
                workList.append([curFile_path, curKey, curSrc, curTrg])
    print("total works: {}".format(len(workList)))
    return workList

for eachTest in testDirs:
    eachTestDir = './outputs_experiment_sn_4spk/'+eachTest
    modelNum = eachTest.split('.')[0].split('_')[-1]
    resultFileName = resultFileName_base+eachTest+'_epoch_'+modelNum+'.txt'
    if os.path.isdir(eachTestDir) == True:
        workList = Make_workList(eachTestDir)
        print("{} MSD calculating start.".format(eachTest))
        with open(resultFileName, 'a') as output:
            p = Pool(30)
            output.writelines(p.map(calculateMCD_mfcc, workList))
            output.writelines("\n")
        p.close()
        print("{} MSD calculating done.".format(eachTest))



#cal average
dir_mcdResults = outFolder

resultFile = open('avg_msd.txt', 'w')
resultFile.write('testName uniqName epoch MSD\n')
for eachFile in os.listdir(dir_mcdResults): #ex) MCD_result_test1_epoch_50
    epoch = eachFile.split('_')[-1].replace('.txt','')
    testName = eachFile
    print('model:{} epoch:{} data calculating...'.format(testName, epoch))
    if 'txt' in eachFile:
        data = open(os.path.join(dir_mcdResults,eachFile),'r').readlines()
        mcd_list = []
        for eachLine in data: #ex) convType: sptp file: t09_s09 F2 -> M2 MCD: 10.476752754383911
            if eachLine != '\n':
                #print('eachLine: <{}>'.format(eachLine))
                _, _, src, _, trg, _, mcd = eachLine.split(' ')
                #print(' MCD:<{}>'.format(mcd))
                mcd = float(mcd.replace('\n',''))
                mcd_list.append(mcd)
        
        #cal avg
        mcd_list = np.average(mcd_list)
        resultFile.write('{} {} {} \n'.format(testName, epoch, mcd_list))
