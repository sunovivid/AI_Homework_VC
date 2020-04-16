# fastdtw for seoul narrative data / modified vaw-gan code
# trying multi processing and can scan sub dir

# directory structure (original source files)
# -source_folder
# --[sn]
# ---evaluation
# ----[spk]
# -----[spk_id_sn]_[file_name].wav (file_name: t[num]_s[num])

# directory structure (converted files)
# -testFolder
# --epoch_[number]
# ---[spk_src]_to_[psk_trg]
# ----[spk_src]-[psk_trg]-[spk_id_sn]_[file_name].wav

import numpy as np
from scipy.spatial.distance import euclidean
import librosa
import os

from multiprocessing import Pool
import time
import math
import argparse
from dtw import dtw
from utils_world import *

speakers=['F1', 'M1', 'F2', 'M2']

source_dir = '../seoulNarrative_allSpk_v2/testSet'
source_wavs = {}
source_wavs_path = {} #just for checking

mfcc_dim = 36 #36

test_dir = "./outputs_experiment_sn_4spk/CCGAN_sn_4spk_1_1800.ckpt/epoch_1800"

result_file_wav = "MCD_result_wav.txt"
result_file_mfcc = "GV_result_mfcc.txt"

counter = 0
for each_spk in speakers:
    for each_file in os.listdir(os.path.join(source_dir, each_spk)):
        if ".wav" in each_file:
            nameParts = each_file.split('.')[0].split('_')
            pure_file_name = nameParts[-2] + '_' + nameParts[-1]
            source_file_key = each_spk + '_' + pure_file_name
            cur_wav_file_loc = os.path.join(source_dir, each_spk, each_file)
            cur_wav_data, _ = librosa.load(cur_wav_file_loc, sr = 16000, mono = True)
            source_wavs[source_file_key] = cur_wav_data
            source_wavs_path[source_file_key] = cur_wav_file_loc
            #print(cur_wav_file_loc)
            counter+=1

#print('total source file: {}'.format(counter))

#dist = lambda x,y: np.linalg.norm(x-y)
dist = lambda x,y: np.linalg.norm((x-y))


ori_whole_array=[]
base_whole_array=[]
def calculateMCD_manual_mfcc(target_converted_file_path, compare_original_file_key, target_converted_source_id, compare_original_target_id):
    if target_converted_source_id == compare_original_target_id:
      return ""
    
    else:
    
      _, nameA, nameB = compare_original_file_key.split('_')
      pure_file_name =  nameA + '_' + nameB
      
      #DTW
      target_conv_wav_data, sr = librosa.load(target_converted_file_path, sr = 22050, mono = True)
      #target_conv_wav_mfcc = librosa.feature.mfcc(y=target_conv_wav_data, sr=sr, n_mfcc=mfcc_dim).T
      #compare_ori_wav_mfcc = librosa.feature.mfcc(y=source_wavs[compare_original_file_key], sr=sr, n_mfcc=mfcc_dim).T
      #distance, path = fastdtw(compare_ori_wav_mfcc, target_conv_wav_mfcc, dist=euclidean)
  
      #MCD
      _, _, _, _,sp_target = world_decompose(wav = target_conv_wav_data, fs = sr, frame_period = 5.0)
      #target_conv_wav_mfcc = world_encode_spectral_envelop(sp = sp_target, fs = sr, dim = mfcc_dim)
      target_conv_wav_mfcc = sp_target
      _, _, _, _,sp_com_ori = world_decompose(wav = source_wavs[compare_original_file_key], fs = sr, frame_period = 5.0)
      print(compare_original_file_key)
      #compare_ori_wav_mfcc = world_encode_spectral_envelop(sp = sp_com_ori, fs = sr, dim = mfcc_dim)
      compare_ori_wav_mfcc = sp_com_ori
      
      base_whole_array.append(np.var(target_conv_wav_mfcc.T,axis=1).tolist())
      ori_whole_array.append(np.var(compare_ori_wav_mfcc.T,axis=1).tolist())
      print(len(base_whole_array))
      #print(len(ori_whole_array))
      
      #whole_array = np.asarray(whole_array)
      
        #whole_array= whole_array.tolist()
      sp_target = np.mean(np.var(target_conv_wav_mfcc.T,axis=1))
      
      sp_com_ori = np.mean(np.var(compare_ori_wav_mfcc.T,axis=1))
      
      #distance, path = fastdtw(compare_ori_wav_mfcc[:,1:], target_conv_wav_mfcc[:,1:], radius = 1000000, dist = dist)
      #mcd = (10.0 / math.log(10)) * math.sqrt(2)* distance / len(path)
      resultLine = "epoch: 1800" + " file: " + pure_file_name + ' ' + target_converted_source_id + " -> " + compare_original_target_id + " GV: " + str(sp_target)+"\n"
      print("{} {} {} {} {}".format('1800', pure_file_name, target_converted_source_id, compare_original_target_id,str(sp_target)))
      return resultLine
    
    

def calculateMCD_mfcc(workData):
    #[target_converted_file_path, compare_original_file_key, model_epoch, target_converted_source_id, compare_original_target_id]
    target_converted_file_path = workData[0]
    compare_original_file_key = workData[1]
    target_converted_source_id = workData[2]
    compare_original_target_id = workData[3]
    return calculateMCD_manual_mfcc(target_converted_file_path, compare_original_file_key, target_converted_source_id, compare_original_target_id)


workList = [] #[target_converted_file_path, compare_original_file_key, model_epoch, target_converted_source_id, compare_original_target_id]
#for each_dir_model in os.listdir(test_dir):
    #cur_model_epoch = each_dir_model.split('_')[-1] #extract model epoch info: epoch_[num]
for each_dir_conv_way in os.listdir(test_dir):
        #print(each_dir_conv_way)
        cur_src, _, cur_trg = each_dir_conv_way.split('_') #extract src/trg info: [src]_to_[trg]
        for each_file in os.listdir(os.path.join(test_dir, each_dir_conv_way)):
            if ".wav" in each_file: #[spk_src]_to_[psk_trg]_[spk_id_sn]_[file_name].wav
                # ex) F1_to_F2_fv01_t09_s06.wav
                nameParts = each_file.split('.')[0].split('_')
                pure_file_name = nameParts[-2] + '_' + nameParts[-1]
                #print(pure_file_name)
                cur_file_path = os.path.join(test_dir,  each_dir_conv_way, each_file)
                compare_original_file_key = cur_trg + '_' + pure_file_name # [target]_[file_name]
                
                workList.append([cur_file_path, compare_original_file_key, cur_src, cur_trg])

#print("total works: {}".format(len(workList)))

with open(result_file_mfcc, 'a') as output:
    #p = Pool(1)
    output.writelines(map(calculateMCD_mfcc, workList))
    output.writelines("\n")
print("mfcc work done.")

np_ori_whole_array = np.average(ori_whole_array,axis=0)
np.savetxt('ori.txt',np_ori_whole_array)

pro_np_whole_array = np.average(base_whole_array,axis=0)
np.savetxt('baseline.txt',pro_np_whole_array)

