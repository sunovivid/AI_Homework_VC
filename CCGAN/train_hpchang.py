import os
import numpy as np
import argparse
import time
import librosa
import pysptk
from multiprocessing import Pool
from dtw import dtw
from preprocess import *
from model import CCGAN
from fastdtw import fastdtw


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
#os.environ["CUDA_VISIBLE_DEVICES"]="1"



def train(speaker_name_list, num_epochs, train_dir, validation_dir, model_dir, model_name, random_seed, output_dir,
          tensorboard_dir='/result/summary', lambda_cycle=10, lambda_identity=5, lambda_adversarial=1):
    print(speaker_name_list)

    np.random.seed(random_seed)
    num_speakers = len(speaker_name_list)

    num_sentences = 81
    num_epochs = num_epochs  # step = num_sentences * (num_speakers ** 2) * num_epochs (81 * 16 * num_epochs)
    no_decay = num_epochs * num_sentences * (num_speakers ** 2) / 2.0
    id_mappig_loss_zero = 10000 * (num_speakers ** 2) # can be less

    mini_batch_size = 1  # must be 1
    generator_learning_rate = 0.0002

    generator_learning_rate_decay = generator_learning_rate / no_decay
    discriminator_learning_rate = 0.0001
    discriminator_learning_rate_decay = discriminator_learning_rate / no_decay
    sampling_rate = 22050
    num_mcep = 36
    frame_period = 5.0
    n_frames = 128
    alpha = pysptk.util.mcepalpha(sampling_rate)  # 0.455
    fftlen = pyworld.get_cheaptrick_fft_size(sampling_rate)

    lambda_cycle = lambda_cycle
    lambda_identity = lambda_identity
    lambda_A2B = lambda_adversarial  # 1 is bad, 3~8 is good

    npz_train_dir = 'VCC2018/npz/vcc2018_training'

    dic_mceps = {}
    # dic_sps_mean = {}
    # dic_sps_std = {}
    dic_f0s_mean = {}
    dic_f0s_std = {}

    print('Preprocessing Data...')

    start_time = time.time()

    for speaker in speaker_name_list:
        if not os.path.exists(os.path.join(npz_train_dir, speaker + '.npz')):
            wavs = load_wavs(wav_dir=os.path.join(train_dir, speaker), sr=sampling_rate)
            f0s, timeaxes, sps, aps, mceps = world_encode_data(wavs=wavs, fs=sampling_rate,
                                                               frame_period=frame_period, coded_dim=num_mcep)
            log_f0s_mean, log_f0s_std = logf0_statistics(f0s)
            # coded_sps_transposed = transpose_in_list(lst=coded_sps)
            # coded_sps_norm, coded_sps_mean, coded_sps_std = coded_sps_normalization_fit_transoform(
            #    coded_sps=coded_sps_transposed)
            np.savez(os.path.join(npz_train_dir, speaker + '.npz'), mceps=mceps, f0s_mean=log_f0s_mean,
                     f0s_std=log_f0s_std)
            # dic_sps_mean[speaker] = coded_sps_mean
            # dic_sps_std[speaker] = coded_sps_std
            dic_f0s_mean[speaker] = log_f0s_mean
            dic_f0s_std[speaker] = log_f0s_std
        else:
            npload = np.load(os.path.join(npz_train_dir, speaker + '.npz'), allow_pickle=True)
            mceps = npload['mceps']
            # dic_sps_mean[speaker] = npload['sps_mean']
            # dic_sps_std[speaker] = npload['sps_std']
            dic_f0s_mean[speaker] = npload['f0s_mean']
            dic_f0s_std[speaker] = npload['f0s_std']
            # coded_sps_transposed = transpose_in_list(lst=coded_sps)
            # coded_sps_norm, _, _ = coded_sps_normalization_fit_transoform(coded_sps=coded_sps_transposed)

        dic_mceps[speaker] = transpose_in_list(mceps)

        print('Log Pitch', speaker)
        print('Mean: %f, Std: %f' % (dic_f0s_mean[speaker], dic_f0s_std[speaker]))

    mceps_list = []
    for speaker in speaker_name_list:
        mceps_list.append(dic_mceps[speaker])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    end_time = time.time()
    time_elapsed = end_time - start_time

    print('Preprocessing Done.')

    print('Time Elapsed for Data Preprocessing: %02d:%02d:%02d' % (
        time_elapsed // 3600, (time_elapsed % 3600 // 60), (time_elapsed % 60 // 1)))

    model = CCGAN(num_features=num_mcep, num_speakers=num_speakers, log_dir=tensorboard_dir)

    for epoch in range(num_epochs):
        print('Epoch: %d' % epoch) 
        start_time_epoch = time.time()
        n_samples, dataset_src, dataset_tar = sample_train_data_mix_all_mapping(mceps_list, n_frames=n_frames)

        for sentence_idx in range(n_samples):
            for case_src in range(num_speakers):
                for case_tar in range(num_speakers):

                    A_id = [0.] * num_speakers
                    B_id = [0.] * num_speakers

                    A_id[case_src] = 1.0
                    B_id[case_tar] = 1.0

                    mapping_direction = num_speakers * case_src + case_tar

                    num_iterations = (num_speakers * num_speakers * n_samples * epoch) + (
                            num_speakers * num_speakers * sentence_idx) + (num_speakers * case_src) + case_tar

                    if num_iterations > id_mappig_loss_zero:
                        lambda_identity = 0

                    if num_iterations > no_decay:  # iteration 
                        generator_learning_rate = max(0, generator_learning_rate - generator_learning_rate_decay)
                        discriminator_learning_rate = max(0,
                                                          discriminator_learning_rate - discriminator_learning_rate_decay)

                    generator_loss, discriminator_loss = model.train(
                        input_A=np.expand_dims(dataset_src[sentence_idx][mapping_direction], axis=0),
                        input_B=np.expand_dims(dataset_tar[sentence_idx][mapping_direction], axis=0),
                        lambda_cycle=lambda_cycle, lambda_identity=lambda_identity, lambda_A2B=lambda_A2B,
                        generator_learning_rate=generator_learning_rate,
                        discriminator_learning_rate=discriminator_learning_rate, A_id=A_id, B_id=B_id)

                    if sentence_idx == 40 or sentence_idx == 80:
                        print(
                            'Epoch: {:04d} Iteration: {:010d}, Generator Learning Rate: {:.8f}, Discriminator Learning Rate: {:.8f}'.format(
                                epoch, num_iterations, generator_learning_rate, discriminator_learning_rate))
                        print(
                            'src: {:s}, tar: {:s}, sent: {:d}, Generator Loss : {:.10f}, Discriminator Loss : {:.10f}'.format(
                                speaker_name_list[case_src], speaker_name_list[case_tar], sentence_idx, generator_loss,
                                discriminator_loss))

        end_time_epoch = time.time()
        time_elapsed_epoch = end_time_epoch - start_time_epoch

        print('Time Elapsed for This Epoch: %02d:%02d:%02d' % (
            time_elapsed_epoch // 3600, (time_elapsed_epoch % 3600 // 60), (time_elapsed_epoch % 60 // 1)))

    model.save(directory=model_dir, filename=model_name) 

    for src in speaker_name_list:
        for tar in speaker_name_list:
            if src == tar:
                continue
            convert_final(speaker_name_list, src, tar, dic_f0s_mean, dic_f0s_std, sampling_rate, num_mcep,
                          frame_period, alpha, fftlen, model)

    calculate_mcd_msd(alpha, 2048, sampling_rate, num_mcep, frame_period, './result/output',
                      './data/speakers_test')


def calculate_mcd_msd(alpha, fftsize, sr=22050, num_mcep=36, frame_period=5.0, validation_dir='./result/output',
                      gt_dir='./data/speakers_test'):
    converted_dirs = os.listdir(validation_dir)
    total_mcd_list = list()
    total_sample_mcd_list = list()
    f2f_mcd_list = list()
    f2m_mcd_list = list()
    m2f_mcd_list = list()
    m2m_mcd_list = list()
    sample_f2f_mcd_list = list()
    sample_f2m_mcd_list = list()
    sample_m2f_mcd_list = list()
    sample_m2m_mcd_list = list()
    total_msd_list = list()
    total_sample_msd_list = list()
    f2f_msd_list = list()
    f2m_msd_list = list()
    m2f_msd_list = list()
    m2m_msd_list = list()
    sample_f2f_msd_list = list()
    sample_f2m_msd_list = list()
    sample_m2f_msd_list = list()
    sample_m2m_msd_list = list()

    for converted_dir in converted_dirs:

        converted_mcep_list = list()
        target_mcep_list = list()
        converted_ms_list = list()
        target_ms_list = list()

        src, tar = converted_dir.split('_to_')
        converted_dir = os.path.join(validation_dir, converted_dir)
        target_dir = os.path.join(gt_dir, tar)
        sent_list = os.listdir(converted_dir)
        for sent in sent_list:
            converted_wav, _ = librosa.load(os.path.join(converted_dir, sent), sr=sr, mono=True)
            _, _, converted_sp, _ = world_decompose(wav=converted_wav, fs=sr, frame_period=frame_period)
            converted_mcep = pysptk.sp2mc(converted_sp, num_mcep - 1, alpha)
            converted_mcep_list.append(converted_mcep[:, 1:])
            converted_ms = logpowerspec(fftsize, converted_mcep)[:1024,:]
            converted_ms_list.append(converted_ms)

            target_wav, _ = librosa.load(os.path.join(target_dir, sent), sr=sr, mono=True)
            _, _, target_sp, _ = world_decompose(wav=target_wav, fs=sr, frame_period=frame_period)
            target_mcep = pysptk.sp2mc(target_sp, num_mcep - 1, alpha)
            target_mcep_list.append(target_mcep[:, 1:])
            target_ms = logpowerspec(fftsize, target_mcep)[:1024,:]
            target_ms_list.append(target_ms)

        p = Pool(35)
        mcd_list = p.starmap(mcd_cal, zip(converted_mcep_list, target_mcep_list))
        msd_list = p.starmap(msd_cal, zip(converted_ms_list, target_ms_list))
        p.close()
        p.join()

        for mcd, msd, sent in zip(mcd_list, msd_list, sent_list):
            total_mcd_list.append(mcd)
            total_msd_list.append(msd)
            if 'F' in src and 'F' in tar:
                f2f_mcd_list.append(mcd)
                f2f_msd_list.append(msd)
                if src == 'VCC2SF1' and tar == 'VCC2SF2' and sent in ['30011.wav', '30012.wav', '30013.wav']:
                    sample_f2f_mcd_list.append(mcd)
                    sample_f2f_msd_list.append(msd)
                    total_sample_mcd_list.append(mcd)
                    total_sample_msd_list.append(msd)
            elif 'F' in src and 'M' in tar:
                f2m_mcd_list.append(mcd)
                f2m_msd_list.append(msd)
                if src == 'VCC2SF2' and tar == 'VCC2SM2' and sent in ['30005.wav', '30009.wav', '30011.wav']:
                    sample_f2m_mcd_list.append(mcd)
                    sample_f2m_msd_list.append(msd)
                    total_sample_mcd_list.append(mcd)
                    total_sample_msd_list.append(msd)

            elif 'M' in src and 'F' in tar:
                m2f_mcd_list.append(mcd)
                m2f_msd_list.append(msd)
                if src == 'VCC2SM1' and tar == 'VCC2SF1' and sent in ['30004.wav', '30005.wav', '30016.wav']:
                    sample_m2f_mcd_list.append(mcd)
                    sample_m2f_msd_list.append(msd)
                    total_sample_mcd_list.append(mcd)
                    total_sample_msd_list.append(msd)

            elif 'M' in src and 'M' in tar:
                m2m_mcd_list.append(mcd)
                m2m_msd_list.append(msd)
                if src == 'VCC2SM2' and tar == 'VCC2SM1' and sent in ['30005.wav', '30009.wav', '30012.wav']:
                    sample_m2m_mcd_list.append(mcd)
                    sample_m2m_msd_list.append(msd)
                    total_sample_mcd_list.append(mcd)
                    total_sample_msd_list.append(msd)

        print('SRC: ', src, ' TAR: ', tar, ' MCD_MEAN: ', np.mean(mcd_list), ' MCD_STD: ', np.std(mcd_list))
        print('SRC: ', src, ' TAR: ', tar, ' MSD_MEAN: ', np.mean(msd_list), ' MSD_STD: ', np.std(msd_list))

    print()
    print('-' * 30)
    print('total ', ' MCD_MEAN: ', np.mean(total_mcd_list), ' MCD_STD: ', np.std(total_mcd_list))
    print('sample ', ' MCD_MEAN: ', np.mean(total_sample_mcd_list), ' MCD_STD: ', np.std(total_sample_mcd_list))
    print('f2f ', ' MCD_MEAN: ', np.mean(f2f_mcd_list), ' MCD_STD: ', np.std(f2f_mcd_list))
    print('f2m ', ' MCD_MEAN: ', np.mean(f2m_mcd_list), ' MCD_STD: ', np.std(f2m_mcd_list))
    print('m2f ', ' MCD_MEAN: ', np.mean(m2f_mcd_list), ' MCD_STD: ', np.std(m2f_mcd_list))
    print('m2m ', ' MCD_MEAN: ', np.mean(m2m_mcd_list), ' MCD_STD: ', np.std(m2m_mcd_list))
    print('sample_f2f ', ' MCD_MEAN: ', np.mean(sample_f2f_mcd_list), ' MCD_STD: ', np.std(sample_f2f_mcd_list))
    print('sample_f2m ', ' MCD_MEAN: ', np.mean(sample_f2m_mcd_list), ' MCD_STD: ', np.std(sample_f2m_mcd_list))
    print('sample_m2f ', ' MCD_MEAN: ', np.mean(sample_m2f_mcd_list), ' MCD_STD: ', np.std(sample_m2f_mcd_list))
    print('sample_m2m ', ' MCD_MEAN: ', np.mean(sample_m2m_mcd_list), ' MCD_STD: ', np.std(sample_m2m_mcd_list))

    print()
    print('-' * 30)
    print('total ', ' MSD_MEAN: ', np.mean(total_msd_list), ' MSD_STD: ', np.std(total_msd_list))
    print('sample ', ' MSD_MEAN: ', np.mean(total_sample_msd_list), ' MCD_STD: ', np.std(total_sample_msd_list))
    print('f2f ', ' MSD_MEAN: ', np.mean(f2f_msd_list), ' MSD_STD: ', np.std(f2f_msd_list))
    print('f2m ', ' MSD_MEAN: ', np.mean(f2m_msd_list), ' MSD_STD: ', np.std(f2m_msd_list))
    print('m2f ', ' MSD_MEAN: ', np.mean(m2f_msd_list), ' MSD_STD: ', np.std(m2f_msd_list))
    print('m2m ', ' MSD_MEAN: ', np.mean(m2m_msd_list), ' MSD_STD: ', np.std(m2m_msd_list))
    print('sample_f2f ', ' MSD_MEAN: ', np.mean(sample_f2f_msd_list), ' MSD_STD: ', np.std(sample_f2f_msd_list))
    print('sample_f2m ', ' MSD_MEAN: ', np.mean(sample_f2m_msd_list), ' MSD_STD: ', np.std(sample_f2m_msd_list))
    print('sample_m2f ', ' MSD_MEAN: ', np.mean(sample_m2f_msd_list), ' MSD_STD: ', np.std(sample_m2f_msd_list))
    print('sample_m2m ', ' MSD_MEAN: ', np.mean(sample_m2m_msd_list), ' MSD_STD: ', np.std(sample_m2m_msd_list))


def mcd_cal(converted_mcep, target_mcep):
    twf = estimate_twf(converted_mcep, target_mcep, fast=False)
    converted_mcep_mod = converted_mcep[twf[0]]
    target_mcep_mod = target_mcep[twf[1]]
    mcd = melcd(converted_mcep_mod, target_mcep_mod)
    return mcd


def msd_cal(converted_ms, target_ms):
    msd = np.sqrt(np.mean(np.power((converted_ms - target_ms), 2)))
    return msd


def logpowerspec(fftsize, data):
    # create zero padded data
    T, dim = data.shape
    padded_data = np.zeros((fftsize, dim))
    padded_data[:T] += data

    complex_spec = np.fft.fftn(padded_data, axes=(0, 1))
    logpowerspec = 2 * np.log(np.absolute(complex_spec))  
    
    return logpowerspec


def melcd(array1, array2):
    """Calculate mel-cepstrum distortion
    Calculate mel-cepstrum distortion between the arrays.
    This function assumes the shapes of arrays are same.
    Parameters
    ----------
    array1, array2 : array, shape (`T`, `dim`) or shape (`dim`)
        Arrays of original and target.
    Returns
    -------
    mcd : scala, number > 0
        Scala of mel-cepstrum distortion
    """
    if array1.shape != array2.shape:
        raise ValueError(
            "The shapes of both arrays are different \
            : {} / {}".format(array1.shape, array2.shape))

    if array1.ndim == 2:
        # array based melcd calculation
        diff = array1 - array2
        mcd = 10.0 / np.log(10) \
              * np.mean(np.sqrt(2.0 * np.sum(diff ** 2, axis=1)))
    elif array1.ndim == 1:
        diff = array1 - array2
        mcd = 10.0 / np.log(10) * np.sqrt(2.0 * np.sum(diff ** 2))
    else:
        raise ValueError("Dimension mismatch")

    return mcd


def estimate_twf(orgdata, tardata, distance='melcd', fast=True, otflag=None):
    if distance == 'melcd':
        def distance_func(x, y):
            return melcd(x, y)
    else:
        raise ValueError('other distance metrics than melcd does not support.')

    if otflag is None:
        # use dtw or fastdtw
        if fast:
            _, path = fastdtw(orgdata, tardata, dist=distance_func)
            twf = np.array(path).T
        else:
            _, _, _, twf = dtw(orgdata, tardata, distance_func)

    return twf


def convert_final(speaker_name_list, src_speaker, tar_speaker, dic_f0s_mean, dic_f0s_std, sampling_rate, num_mcep,
                  frame_period, alpha, fftlen, model):
    A = src_speaker
    B = tar_speaker
    num_speakers = len(speaker_name_list)

    A_id_v = [0.] * num_speakers
    B_id_v = [0.] * num_speakers

    A_index = speaker_name_list.index(A)
    B_index = speaker_name_list.index(B)

    A_id_v[A_index] = 1.0
    B_id_v[B_index] = 1.0

    log_f0s_mean_A = dic_f0s_mean[A]
    log_f0s_std_A = dic_f0s_std[A]
    log_f0s_mean_B = dic_f0s_mean[B]
    log_f0s_std_B = dic_f0s_std[B]

    if validation_dir is not None:
        validation_A_dir = os.path.join(validation_dir, A)

    if validation_A_dir is not None:
        validation_A_output_dir = os.path.join(output_dir, A + '_to_' + B)
        if not os.path.exists(validation_A_output_dir):
            os.makedirs(validation_A_output_dir)

    print('validaiton (SRC: %s, TAR: %s)' % (A, B))
    for file in os.listdir(validation_A_dir):
        filepath = os.path.join(validation_A_dir, file)
        wav, _ = librosa.load(filepath, sr=sampling_rate, mono=True)
        wav = wav_padding(wav=wav, sr=sampling_rate, frame_period=frame_period, multiple=4)
        f0, timeaxis, sp, ap = world_decompose(wav=wav, fs=sampling_rate, frame_period=frame_period)
        f0_converted = pitch_conversion(f0=f0, mean_log_src=log_f0s_mean_A, std_log_src=log_f0s_std_A,
                                        mean_log_target=log_f0s_mean_B, std_log_target=log_f0s_std_B)

        mcep = pysptk.sp2mc(sp, num_mcep - 1, alpha)
        mcepT = mcep.T
        converted_mcep = model.test(inputs=np.array([mcepT]), A_id=A_id_v, B_id=B_id_v)[0]
        converted_mcepT = converted_mcep.T

        sp_converted = pysptk.mc2sp(converted_mcepT, alpha, fftlen)
        sp_converted = sp_converted.astype('float64')
        wav_transformed = world_speech_synthesis(f0=f0_converted, decoded_sp=sp_converted, ap=ap,
                                                 fs=sampling_rate, frame_period=frame_period)
        librosa.output.write_wav(os.path.join(validation_A_output_dir, os.path.basename(file)), wav_transformed,
                                 sampling_rate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CCGAN model for datasets.')

    speaker_name_list_default = ['VCC2SF1', 'VCC2SF2', 'VCC2SM1', 'VCC2SM2']
    num_epochs_default = 200
    train_dir_default = './data/speakers'
    model_dir_default = './result/model'
    model_name_default = 'CCGAN.ckpt'
    random_seed_default = 0
    validation_dir_default = './data/speakers_test'
    output_dir_default = './result/output'
    tensorboard_log_dir_default = './result/summary'

    parser.add_argument('-speaker_name_list', type=str, nargs='+', help='Speaker Name List.',
                        default=speaker_name_list_default)
    parser.add_argument('-num_epochs', type=int, help='Number of Epoch.', default=num_epochs_default)
    parser.add_argument('-train_dir', type=str, help='Directory for training.', default=train_dir_default)
    parser.add_argument('-validation_dir', type=str,
                        help='If set none, no conversion would be done during the training.',
                        default=validation_dir_default)
    parser.add_argument('-model_dir', type=str, help='Directory for saving models.', default=model_dir_default)
    parser.add_argument('-model_name', type=str, help='File name for saving model.', default=model_name_default)
    parser.add_argument('-random_seed', type=int, help='Random seed for model training.', default=random_seed_default)
    parser.add_argument('-output_dir', type=str, help='Output directory for converted validation voices.',
                        default=output_dir_default)
    parser.add_argument('-tensorboard_dir', type=str, help='TensorBoard log directory.',
                        default=tensorboard_log_dir_default)
    parser.add_argument('-lambda_cycle', type=int, help='Weight of Cycle Loss',
                        default=10)
    parser.add_argument('-lambda_identity', type=int, help='Weight of Identity Mapping Loss',
                        default=5)
    parser.add_argument('-lambda_A2B', type=int, help='Weight of Adversarial Loss',
                        default=3)

    argv = parser.parse_args()

    speaker_name_list = argv.speaker_name_list
    print(speaker_name_list)
    num_epochs = argv.num_epochs
    train_dir = argv.train_dir
    model_dir = argv.model_dir
    model_name = argv.model_name
    random_seed = argv.random_seed
    validation_dir = None if argv.validation_dir == 'None' or argv.validation_dir == 'none' else argv.validation_dir
    output_dir = argv.output_dir
    tensorboard_dir = argv.tensorboard_dir
    lambda_cycle = argv.lambda_cycle
    lambda_identity = argv.lambda_identity
    lambda_A2B = argv.lambda_A2B

    train(speaker_name_list, num_epochs, train_dir, validation_dir, model_dir, model_name, random_seed, output_dir,
          tensorboard_dir, lambda_cycle, lambda_identity, lambda_A2B)
