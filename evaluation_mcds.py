import argparse
from mel_cepstral_distance import get_metrics_wavs, get_metrics_mels
import os
import csv
import glob
import numpy as np
import wave
import pyworld
import librosa
from tqdm import tqdm
from os.path import basename
from utils import *

def get_wavfiles_dict(speakers_list, parent_directory):
    # Create an empty dictionary to store the results
    wav_files_dict = {}

    # Iterate through the list of speakers
    for speaker_id in speakers_list:
        # Construct the path to the speaker's subfolder
        speaker_folder = os.path.join(parent_directory, speaker_id)

        # Check if the folder exists
        if os.path.exists(speaker_folder) and os.path.isdir(speaker_folder):
            # List all the WAV files in the speaker's subfolder
            wav_files = [f for f in os.listdir(speaker_folder) if f.endswith(".wav")]

            # Store the list of WAV files in the dictionary with the speaker ID as the key
            wav_files_dict[speaker_id] = wav_files
        else:
            print(f"Folder not found for speaker {speaker_id}")

    return wav_files_dict

def get_convertedwavfiles_dict(speakers_list, parent_directory):
    converted_wav_files_dict = {}

    # List all the WAV files in the parent directory
    wav_files = [f for f in os.listdir(parent_directory) if f.endswith(".wav")]

    tmp_dict = {}
    # Iteratre over the WAV files
    for wav_file in wav_files:
        if "cpsyn" not in wav_file:
            wav_file_split = wav_file.split("-")
            iteration_n = wav_file_split[0]
            org_spk = wav_file_split[1][:4]
            target_spk = wav_file_split[-1][:4]

            # Check if iteration_n key exists in tmp_dict
            if iteration_n in tmp_dict:
                # Append the wav_file to the existing list
                tmp_dict[iteration_n].append(wav_file)
            else:
                # Create a new list and add the wav_file to it
                tmp_dict[iteration_n] = [wav_file]

            converted_wav_files_dict[org_spk + "-vcto-" + target_spk] = tmp_dict


    return  converted_wav_files_dict


def main(config):

    # Directories
    if config.where_exec == "slurm":
        config.gnrl_data_dir = '/workspace/NASFolder'
        output_directory = os.path.join(config.gnrl_data_dir, "output")
    elif config.where_exec == "local":
        config.gnrl_data_dir = "E:/TFM_EN_ESTE_DISCO_DURO/TFM_project/"

    # Get original data directory
    original_wavs_dir = os.path.join(config.gnrl_data_dir, "data/wav16") #)"data/mcs"
    original_wavs_files_dict = get_wavfiles_dict(speakers_list=config.speakers, parent_directory=original_wavs_dir)

    # Get converted data directory
    converted_wavs_dir = config.converted_samples_data_dir
    converted_wavs_files_dict = get_convertedwavfiles_dict(speakers_list=config.speakers, parent_directory=converted_wavs_dir)

    # Test
    original_wavfile_path = os.path.join(original_wavs_dir, "p272", original_wavs_files_dict["p272"][5])
    converted_wavfile_path = os.path.join(converted_wavs_dir, converted_wavs_files_dict["p262-vcto-p272"]["10000"][0])

    mcd, penalty, frames = get_metrics_wavs(wav_file_1=original_wavfile_path, wav_file_2=converted_wavfile_path)

    original_wavs_files = ""




    # Convereted da



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Directories.
    parser.add_argument('--gnrl_data_dir', type=str, default='.')
    parser.add_argument('--converted_samples_data_dir', type=str, default="Z:/Shared_PFC-TFG-TFM/Claudio/TOPK_VC/output/samples/[30_09_2023]_TopK_v2/sgv_v2_topk_False_topk_g_0.9999_topk_v_0.5_topk_fi_25000_lbd_rec_10.0_lbd_rec_5.0_lambda_id_5.0_lbd_cls_10.0_sr_16000_bs_32_iters_200000_iters_dec_100000_g_lr_0.0001_d_lr_0.0001_n_critic_5_b1_0.5_b2_0.999")
    parser.add_argument('--where_exec', type=str, default='local', help="slurm or local") # "slurm", "local"
    parser.add_argument('--speakers', type=str, nargs='+', required=False, help='Speaker dir names.',
                        default=["p272", "p262", "p232", "p229"])

    config = parser.parse_args()
    main(config)

spks = ["p272", "p262", "p232", "p229"]

original_wavs_dir = ""


trg_dir = "/content/drive/MyDrive/audio_samples/target_original_samples"
convert_dir = "/content/drive/MyDrive/audio_samples/sb_vanilla_samples"
trg = "p272"
spk_to_spk = "p262_to_p272"
output_csv = "p262_to_p272.csv"

trg_files = glob.glob(os.path.join(trg_dir, '*.wav'))
vcto_files = glob.glob(os.path.join(convert_dir, '*' + '-vcto-'+ '*.wav'))
print(vcto_files, trg_files)

sample_rate = get_sampling_rate(trg_files[0])
print(sample_rate)

get_spk_world_feats(trg, trg_files, os.path.join(convert_dir, spk_to_spk), sample_rate)
get_spk_world_feats('vcto', vcto_files, os.path.join(convert_dir, spk_to_spk), sample_rate)

trg_npy_files = glob.glob(os.path.join(convert_dir, spk_to_spk, trg + '*.npy'))
vcto_npy_files = glob.glob(os.path.join(convert_dir, spk_to_spk, '*-vcto-*.npy'))

trg_npy_files.sort()
vcto_npy_files.sort()

print("MCD OF SAM BROUGHTON VANILLA")

with open(os.path.join(convert_dir, output_csv), 'wt') as csv_f:
  csv_w = csv.writer(csv_f, delimiter=',')
  csv_w.writerow(['SPK_to_SPK', 'REFERENCE', 'SYNTHESIZED', 'MCD'])
  for idx, ref in enumerate(trg_npy_files):
    synth = vcto_npy_files[idx]
    dist = mel_cep_dtw_dist(np.load(ref, allow_pickle=True), np.load(synth, allow_pickle=True))
    print(f'MCD | {ref} to {synth} = {dist}')
    csv_w.writerow([spk_to_spk, os.path.basename(ref), os.path.basename(synth), dist])