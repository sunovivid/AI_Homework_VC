import os
import numpy as np
import argparse
import time
import librosa

from preprocess import *
from model import CCGAN

import soundfile as sf #insert: for correct wav writing


def train(speaker_name_list, num_epochs, train_dir, validation_dir, model_dir, model_name, random_seed, output_dir, tensorboard_log_dir, text_log_dir):
    
    print(speaker_name_list)
    
    np.random.seed(random_seed)
    num_speakers = len(speaker_name_list)
    
    num_sentences = 200 # min(number of setences for each speaker)
    num_epochs = num_epochs
    no_decay = num_epochs * num_sentences * (num_speakers ** 2) / 2.0
    id_mappig_loss_zero = 10000 * (num_speakers ** 2) # can be less
    
    mini_batch_size = 1 # must be 1
    generator_learning_rate = 0.0002
    
    generator_learning_rate_decay = generator_learning_rate / no_decay
    discriminator_learning_rate = 0.0001
    discriminator_learning_rate_decay = discriminator_learning_rate / no_decay
    sampling_rate = 16000
    num_mcep = 24
    frame_period = 5.0
    n_frames = 128
    
    lambda_cycle = 10
    lambda_identity = 5
    lambda_A2B = 3 # 1 is bad, 3~8 is good
    
    dic_sps_norm = {}
    dic_sps_mean = {}
    dic_sps_std = {}
    dic_f0s_mean = {}
    dic_f0s_std = {}

    print('Preprocessing Data...')

    start_time = time.time()
    
    for speaker in speaker_name_list:
        if not os.path.exists(os.path.join(model_dir, speaker + '.npz')):
            wavs = load_wavs(wav_dir = os.path.join(train_dir, speaker), sr = sampling_rate)
            f0s, timeaxes, sps, aps, coded_sps = world_encode_data(wavs = wavs, fs = sampling_rate, frame_period = frame_period, coded_dim = num_mcep)
            log_f0s_mean, log_f0s_std = logf0_statistics(f0s)
            coded_sps_transposed = transpose_in_list(lst = coded_sps)
            coded_sps_norm, coded_sps_mean, coded_sps_std = coded_sps_normalization_fit_transoform(coded_sps = coded_sps_transposed)
            np.savez(os.path.join(model_dir, speaker + '.npz'), coded_sps = coded_sps, f0s_mean = log_f0s_mean, f0s_std = log_f0s_std, sps_mean = coded_sps_mean, sps_std = coded_sps_std)
            dic_sps_mean[speaker] = coded_sps_mean
            dic_sps_std[speaker] = coded_sps_std
            dic_f0s_mean[speaker] = log_f0s_mean
            dic_f0s_std[speaker] = log_f0s_std
            del wavs
        else:    
            npload = np.load(os.path.join(model_dir, speaker + '.npz'), allow_pickle = True)
            coded_sps = npload['coded_sps']
            dic_sps_mean[speaker] = npload['sps_mean']
            dic_sps_std[speaker] = npload['sps_std']
            dic_f0s_mean[speaker] = npload['f0s_mean']
            dic_f0s_std[speaker] = npload['f0s_std']
            coded_sps_transposed = transpose_in_list(lst = coded_sps)
            coded_sps_norm, _, _ = coded_sps_normalization_fit_transoform(coded_sps = coded_sps_transposed)
        
        dic_sps_norm[speaker] = coded_sps_norm 
        
        print('Log Pitch', speaker)
        print('Mean: %f, Std: %f' %(dic_f0s_mean[speaker], dic_f0s_std[speaker]))
    
    '''coded_sps_norm_list = []
    for speaker in speaker_name_list:
        coded_sps_norm_list.append(dic_sps_norm[speaker])'''
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    end_time = time.time()
    time_elapsed = end_time - start_time

    print('Preprocessing Done.')

    print('Time Elapsed for Data Preprocessing: %02d:%02d:%02d' % (time_elapsed // 3600, (time_elapsed % 3600 // 60), (time_elapsed % 60 // 1)))

    model = CCGAN(num_features = num_mcep, num_speakers = num_speakers)

    if not os.path.exists(text_log_dir):
        os.makedirs(text_log_dir)

    total_train_time = 0
    for epoch in range(num_epochs):
        print('Epoch: %d' % epoch) # 현재 에포크 출력

        start_time_epoch = time.time()
        #n_samples, dataset_src, dataset_tar = sample_train_data_mix_all_mapping(coded_sps_norm_list, n_frames = 128)
        # out shape: (num_sentences, mapping_direction?, n_frames, mfcc_dim)
               
        for sentence_idx in range(num_sentences):
            for case_src in range(num_speakers):
                for case_tar in range(num_speakers):
                    src_id = speaker_name_list[case_src]
                    tar_id = speaker_name_list[case_tar]
                    #print('src:{} tar:{} sen:{}/{}'.format(src_id, tar_id, sentence_idx+1, num_sentences))

                    dataset_src, dataset_tar = sample_train_data_mix_mapping(dic_sps_norm[src_id], dic_sps_norm[tar_id], num_sentences, n_frames = 128)
                    # out shape: (num_sentences, mfcc_dim, n_frames)
                    A_id = [0.]*num_speakers
                    B_id = [0.]*num_speakers
                    
                    A_id[case_src] = 1.0
                    B_id[case_tar] = 1.0
                    
                    mapping_direction = num_speakers * case_src + case_tar

                    num_iterations = (num_speakers * num_speakers * num_sentences * epoch) + (num_speakers * num_speakers * sentence_idx) + (num_speakers * case_src) + case_tar

                    if num_iterations > id_mappig_loss_zero:
                        lambda_identity = 0
                        
                    if num_iterations > no_decay: # iteration 이 넘어가면 선형감쇠 = 점점 조금 학습
                        generator_learning_rate = max(0, generator_learning_rate - generator_learning_rate_decay)
                        discriminator_learning_rate = max(0, discriminator_learning_rate - discriminator_learning_rate_decay)
                        
                    generator_loss, discriminator_loss = model.train(input_A = np.expand_dims(dataset_src[sentence_idx], axis = 0), input_B = np.expand_dims(dataset_tar[sentence_idx], axis = 0), lambda_cycle = lambda_cycle, lambda_identity = lambda_identity, lambda_A2B = lambda_A2B, generator_learning_rate = generator_learning_rate, discriminator_learning_rate = discriminator_learning_rate, A_id = A_id, B_id = B_id)

                    if sentence_idx == num_sentences-1:
                        print('Epoch: {:04d} Iteration: {:010d}, Generator Learning Rate: {:.8f}, Discriminator Learning Rate: {:.8f}'.format(epoch, num_iterations, generator_learning_rate, discriminator_learning_rate))
                        print('src: {:s}, tar: {:s}, sent: {:d}, Generator Loss : {:.10f}, Discriminator Loss : {:.10f}'.format(speaker_name_list[case_src], speaker_name_list[case_tar], sentence_idx, generator_loss, discriminator_loss))
                        with open(text_log_dir+'/losses.txt','a') as log_file:
                            log_file.write('src: {:s} tar: {:s} sent: {:d} Generator_Loss: {:.10f} Discriminator_Loss: {:.10f}\n'.format(speaker_name_list[case_src], speaker_name_list[case_tar], sentence_idx+1, generator_loss, discriminator_loss))

        if epoch % 1 == 0 and epoch != 0:
            model.save(directory = model_dir, filename = model_name+'_'+str(epoch)) #모델저장

        end_time_epoch = time.time()
        time_elapsed_epoch = end_time_epoch - start_time_epoch
        print('Time Elapsed for This Epoch: %02d:%02d:%02d' % (time_elapsed_epoch // 3600, (time_elapsed_epoch % 3600 // 60), (time_elapsed_epoch % 60 // 1)))
        total_train_time += time_elapsed_epoch
       
        # validaiton is conducted for first speaker and last speaker of the speaker list
        # if you want full mapping validation, run convert_all.py
        
        if (epoch % 1 == 0 or epoch == num_epochs - 1) and epoch != 0:
            convert(speaker_name_list, model_dir, model, validation_dir, output_dir, epoch, max_convert=1) #for intertest, just convert 1 file.
        
    convert_time = '{}:{}:{}'.format(int(total_train_time // 3600), int(total_train_time % 3600 // 60), int(total_train_time % 60 // 1))
    print('total network training time: {}'.format(convert_time))
    with open(text_log_dir+'/losses.txt','a') as log_file:
        log_file.write('total_network_training_time: {}'.format(convert_time))

    model.save(directory = model_dir, filename = model_name+'_'+str(epoch)) #final model save
        
def convert(speaker_name_list, model_dir, model, data_dir, output_dir, epoch, max_convert=None):
    start_time = time.time()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    num_features = 24
    sampling_rate = 16000
    frame_period = 5.0
        
    num_speakers = len(speaker_name_list)
    
    dic_sps_mean = {}
    dic_sps_std = {}
    dic_f0s_mean = {}
    dic_f0s_std = {}

    for speaker in speaker_name_list:
        npload = np.load(os.path.join(model_dir, speaker + '.npz'), allow_pickle = True)
        dic_sps_mean[speaker] = npload['sps_mean']
        dic_sps_std[speaker] = npload['sps_std']
        dic_f0s_mean[speaker] = npload['f0s_mean']
        dic_f0s_std[speaker] = npload['f0s_std']
    
    completed = 0
    
    for src in speaker_name_list:
        for tar in speaker_name_list:
        
            src_idx = speaker_name_list.index(src)
            tar_idx = speaker_name_list.index(tar)

            src_id = [0.]*num_speakers
            src_id[src_idx] = 1.0

            tar_id = [0.]*num_speakers
            tar_id[tar_idx] = 1.0
            
            out_dir = os.path.join(output_dir, (str(epoch) + 'epoch_' + src + '_to_' + tar))
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            
            print(src + ' -> ' + tar + ' converting...')

            mcep_mean_A = dic_sps_mean[src]
            mcep_std_A = dic_sps_std[src]
            mcep_mean_B = dic_sps_mean[tar]
            mcep_std_B = dic_sps_std[tar]

            logf0s_mean_A = dic_f0s_mean[src]
            logf0s_std_A = dic_f0s_std[src]
            logf0s_mean_B = dic_f0s_mean[tar]
            logf0s_std_B = dic_f0s_std[tar]
            
            data_dir_src = os.path.join(data_dir, src)
            assert os.path.exists(data_dir_src)

            counter = 0
            for file in os.listdir(data_dir_src):
                if max_convert != None: # limited converting
                    counter += 1
                    if counter > max_convert:
                        break
                
                filepath = os.path.join(data_dir_src, file)
                wav, _ = librosa.load(filepath, sr = sampling_rate, mono = True)
                wav = wav_padding(wav = wav, sr = sampling_rate, frame_period = frame_period, multiple = 4)
                f0, timeaxis, sp, ap = world_decompose(wav = wav, fs = sampling_rate, frame_period = frame_period)
                coded_sp = world_encode_spectral_envelop(sp = sp, fs = sampling_rate, dim = num_features)
                coded_sp_transposed = coded_sp.T
            
                f0_converted = pitch_conversion(f0 = f0, mean_log_src = logf0s_mean_A, std_log_src = logf0s_std_A, mean_log_target = logf0s_mean_B, std_log_target = logf0s_std_B)
                coded_sp_norm = (coded_sp_transposed - mcep_mean_A) / mcep_std_A
                coded_sp_converted_norm = model.test(inputs = np.array([coded_sp_norm]), A_id = src_id, B_id = tar_id)[0]
                coded_sp_converted = coded_sp_converted_norm * mcep_std_B + mcep_mean_B

                coded_sp_converted = coded_sp_converted.T
                coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
                decoded_sp_converted = world_decode_spectral_envelop(coded_sp = coded_sp_converted, fs = sampling_rate)
                decoded_sp_converted = decoded_sp_converted[:len(f0_converted),:]#insert
                wav_transformed = world_speech_synthesis(f0 = f0_converted, decoded_sp = decoded_sp_converted, ap = ap, fs = sampling_rate, frame_period = frame_period)
                #librosa.output.write_wav(os.path.join(out_dir, src + '_to_' + tar + '_' + os.path.basename(file)), wav_transformed, sampling_rate)
                sf.write(os.path.join(out_dir, src + '_to_' + tar + '_' + os.path.basename(file)), wav_transformed, sampling_rate, 'PCM_16')#insert
            
            completed += 1
            print(str(completed) + ' /', num_speakers**2, 'conversion completed')
    
    end_time = time.time()
    time_elapsed = end_time - start_time
    
    print('\nConversion Done.')
    print('Time Elapsed for conversion: %02d:%02d:%02d' % (time_elapsed // 3600, (time_elapsed % 3600 // 60), (time_elapsed % 60 // 1)))

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description = 'Train & Test CCGAN model for modified SN datasets.')
    
    '''speaker_name_list_default = []
    # default spk ids
    max_spk_f=59
    max_spk_m=56
    for idx in range(max_spk_f):
        spk_id = 'F'+str(idx+1)
        speaker_name_list_default.append(spk_id)
    for idx in range(max_spk_m):
        spk_id = 'M'+str(idx+1)
        speaker_name_list_default.append(spk_id)'''

    speaker_name_list_default = ['F1','F2','M1','M2']

    num_epochs_default = 251
    train_dir_default = '../seoulNarrative_4spk_v3/trainingSet'
    model_dir_default = './model_sn_4spk'
    model_name_default = 'CCGAN_sn.ckpt'
    random_seed_default = 0
    validation_dir_default = '../seoulNarrative_4spk_v3/testSet'
    output_dir_default = './outputs_experiment_sn'
    tensorboard_log_dir_default = './log'
    text_log_dir_default = './log_txt'
    
    parser.add_argument('--speaker_name_list', type = str, nargs='+', help = 'Speaker Name List.', default = speaker_name_list_default)
    parser.add_argument('--num_epochs', type = int, help = 'Number of Epoch.', default = num_epochs_default)
    parser.add_argument('--train_dir', type = str, help = 'Directory for training.', default = train_dir_default)
    parser.add_argument('--validation_dir', type = str, help = 'If set none, no conversion would be done during the training.', default = validation_dir_default)
    parser.add_argument('--model_dir', type = str, help = 'Directory for saving models.', default = model_dir_default)
    parser.add_argument('--model_name', type = str, help = 'File name for saving model.', default = model_name_default)
    parser.add_argument('--random_seed', type = int, help = 'Random seed for model training.', default = random_seed_default)
    parser.add_argument('--output_dir', type = str, help = 'Output directory for converted validation voices.', default = output_dir_default)
    parser.add_argument('--tensorboard_log_dir', type = str, help = 'TensorBoard log directory.', default = tensorboard_log_dir_default)
    parser.add_argument('--text_log_dir', type = str, help = 'Text log directory.', default = text_log_dir_default)

    argv = parser.parse_args()

    speaker_name_list = argv.speaker_name_list
    num_epochs = argv.num_epochs
    train_dir = argv.train_dir
    model_dir = argv.model_dir
    model_name = argv.model_name
    random_seed = argv.random_seed
    validation_dir = None if argv.validation_dir == 'None' or argv.validation_dir == 'none' else argv.validation_dir
    output_dir = argv.output_dir
    tensorboard_log_dir = argv.tensorboard_log_dir
    text_log_dir = argv.text_log_dir
    
    train(speaker_name_list, num_epochs, train_dir, validation_dir, model_dir, model_name, random_seed, output_dir, tensorboard_log_dir, text_log_dir)      
