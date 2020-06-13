#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pydub import AudioSegment
from tqdm import tqdm
import os
from random import shuffle
from pathlib import Path
import librosa
import numpy as np
import subprocess
from glob import glob
import shutil
import concurrent.futures
from multiprocessing import Pool
from unidecode import unidecode
#from g2p_en import G2p
#g2p = G2p()

# Functions
def force_move_dir(root_src_dir, root_dst_dir):
    for src_dir, dirs, files in tqdm(os.walk(root_src_dir), smoothing=0):
        dst_dir = src_dir.replace(root_src_dir, root_dst_dir, 1)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        for file_ in files:
            src_file = os.path.join(src_dir, file_)
            dst_file = os.path.join(dst_dir, file_)
            if os.path.exists(dst_file):
                # in case of the src and dst are the same file
                if os.path.samefile(src_file, dst_file):
                    continue
                os.remove(dst_file)
            shutil.move(src_file, dst_dir)


def reset_directory_structure(inputpath, outputpath, verbose=True):
    """delete outputpath, copy folder structure of inputpath to outputpath but no files"""
    assert inputpath != outputpath
    import shutil
    if os.path.exists(outputpath):
        if verbose:
            print("Resetting ",outputpath)
        shutil.rmtree(outputpath)
    def ig_f(dir, files):
        return [f for f in files if os.path.isfile(os.path.join(dir, f))]
    shutil.copytree(inputpath, outputpath, ignore=ig_f)


def even_split(a, n):
    """split array into n seperate evenly sized chunks"""
    n = min(n, len(a)) # if less elements in array than chunks to output, change chunks to array length
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    # list(chunks([0,1,2,3,4,5,6,7,8,9],2)) -> [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def get_arpadict(dictionary_path):
    print("Running, Please wait...")
    thisdict = {}
    for line in reversed((open(dictionary_path, "r").read()).splitlines()):
        thisdict[unidecode((line.split(" ",1))[0])] = unidecode((line.split(" ",1))[1].strip())
    print("Dictionary Ready.")
    return thisdict


def concat_text(filenames, outpath):
    with open(outpath, 'w') as outfile:
        nan = 0
        for fname in filenames:
            if nan == 1: outfile.write("\n") # add newlines (\n) between each file
            else: nan = 1
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)


def arpabet(input_path, output_path, start_tokens=True, encoding="utf-8"):
    errored_words = ""
    sym = list(r"""!?,.;:␤"'#-_☺☻♥♦♣♠•◘○◙♂♀♪♫☼►◄↕‼¶§▬↨↑↓→←∟↔▲""") # ␤ = new line
    arr = []
    for line in ((open(input_path, "r").read()).splitlines()):
        out = ''
        for word_ in (line.split("|")[1]).split(" "):
            word=word_; end_chars = ''; start_chars = ''
            while any(elem in word for elem in sym) and len(word) > 1:
                if word[-1] in sym: end_chars = word[-1] + end_chars; word = word[:-1]
                elif word[0] in sym: start_chars = start_chars + word[0]; word = word[1:]
                else: break
            try:
                word = "{" + thisdict[word.upper()] + "}"
            except KeyError:
                pass
            out = (out + " " + start_chars + word + end_chars).strip()
        arr.append(f'{line.split("|")[0]}|{out}␤|{line.split("|")[2]}')
    text_file = open(output_path, "w", encoding=encoding)
    text_file.write("\n".join(arr))
    text_file.close()


def nancy_build_metadata(directory, prompts, SAMPLE_RATE=48000, BIT_DEPTH=2, min_audio_duration=0.8, max_audio_duration=12.0):
    Nancy_lookup = {filename: quote[1:-1].strip() for filename, quote in [line[2:-2].split(" ",1) for line in ((open(prompts, "r").read()).splitlines())]}
    short_clip_count = long_clip_count = total_clip_count = invalid_clip_count = 0
    for audio_file in glob(directory+"/**/*.wav", recursive=True):
        total_clip_count+=1
        if os.stat(audio_file).st_size < int(BIT_DEPTH*SAMPLE_RATE*min_audio_duration): short_clip_count+=1; continue # if file length less than min_audio_duration seconds, skip
        if os.stat(audio_file).st_size > int(BIT_DEPTH*SAMPLE_RATE*max_audio_duration): long_clip_count+=1; continue #  if file length greater than max_audio_duration seconds, skip
        
        audio_filename = "/".join(audio_file.split("/")[-1:])
        if audio_filename.replace(".wav","") in Nancy_lookup.keys():
            quote = Nancy_lookup[audio_filename.replace(".wav","")]
            quote = unidecode(quote)
            timestamp = "00_00_00"
            voice = "(Audiobook) Blizzard2011_"+"Nancy"
            emotions = []
            noise_level = ""
            
            if voice.title() in list(metadata.keys()):
                metadata[str(voice).title()].append({"file_path": audio_file, "timestamp": timestamp, "emotions": emotions, "noise_level": noise_level, "quote": quote})
            else:
                metadata[str(voice).title()] = [{"file_path": audio_file, "timestamp": timestamp, "emotions": emotions, "noise_level": noise_level, "quote": quote}]
        else:
            print(f"{audio_file} Has no Quote.")
    print(str(total_clip_count)+" Total Clips")
    print(str(short_clip_count)+" clips are too short")
    print(str(long_clip_count)+" clips are too long")
    print(str(invalid_clip_count)+" clips are invalid (bad filename or missing quote)")
    print(str(total_clip_count-(short_clip_count+long_clip_count+invalid_clip_count))+" clips written into metadata dict")


# metadata["celestia"] =  [{file_path: "", timestamp: "00_00_05", emotions: ["neutral"], noise_level: "", quote = "Once upon a time."}, .... , ....]
def build_metadata(directory, ignore_dirs=["Noise samples",], SAMPLE_RATE=48000, BIT_DEPTH=2, min_audio_duration=0.8, max_audio_duration=12.0): # uses Global "directory" for recursive directory search, for every .wav file it will find the accompanying label and add to metadata.
    short_clip_count = 0
    long_clip_count = 0
    total_clip_count = 0
    invalid_clip_count = 0
    for dir_ in [x[0] for x in os.walk(directory)]: # recursive directory search
        if len(os.listdir(dir_)) < 1: continue
        for filename in os.listdir(dir_):
            file_path = os.path.join(dir_,filename)
            if any([ignoredir in file_path for ignoredir in ignore_dirs]): continue
            if filename.endswith(".wav"):
                total_clip_count+=1
                if os.stat(file_path).st_size < int(BIT_DEPTH*SAMPLE_RATE*min_audio_duration): short_clip_count+=1; continue # if file length less than min_audio_duration seconds, skip
                if os.stat(file_path).st_size > int(BIT_DEPTH*SAMPLE_RATE*max_audio_duration): long_clip_count+=1; continue #  if file length greater than max_audio_duration seconds, skip
                # ---- Unique DATASETs ----
                if "Anons/PostalDude2" in file_path:
                    timestamp = "00_00_00"
                    voice = "postal dude"
                    emotions = []
                    noise_level = ""
                    filename_quote = None # missing question marks
                elif "Anons/Rise Kujikawa" in file_path:
                    timestamp = "00_00_00"
                    voice = "Rise Kujikawa"
                    emotions = []
                    noise_level = ""
                    filename_quote = None # missing question marks
                # ---- Unique DATASETs ----
                # ---- VCTK and Clipper Sliced DATASET ----
                else:
                    splitted = filename.split("_")
                    try: # if Clippers MLP dataset
                        timestamp = "_".join(splitted[0:3])          # 00_00_05
                        voice = "(Show) My Little Pony_"+splitted[3].title()# e.g: (Show) My Little Pony_Celestia
                        
                        if "Other/Star Trek (John de Lancie, Discord)" in file_path:
                            voice = "(Show) Star Trek_"+"Q"
                        elif "Other/Eli, Elite Dangerous (John de Lancie, Discord)" in file_path:
                            voice = "(Game) Elite Dangerous_"+"Eli"
                        elif "Other/A Little Bit Wicked (Kristin Chenoworth, Skystar)" in file_path:
                            voice = "(Audiobook) A Little Bit Wicked_"+splitted[3].title()
                        elif "Other/Sum - Tales From the Afterlives (Emily Blunt, Tempest)" in file_path:
                            voice = "(Audiobook) Sum - Tales From the Afterlives_"+splitted[3].title()
                        elif "Other/Dr. Who" in file_path:
                            voice = "(Audiobook) Dr. Who_"+splitted[3].title()
                        elif "Other/Dan vs" in file_path:
                            voice = "(Show) Dan vs_"+splitted[3].title()
                        elif "Other/TFH" in file_path:
                            voice = "(Game) Them's Fightin' Herds_"+splitted[3].title()
                        elif "/FoE s1e01 Radioplay" in file_path:
                            voice = "(Audiodrama) Fallout Equestria_"+splitted[3].title()
                        elif "/FoE s1e02 Radio Play" in file_path:
                            voice = "(Audiodrama) Fallout Equestria_"+splitted[3].title()
                        elif "/Songs" in file_path:
                            voice = "(Music) My Little Pony_"+splitted[3].title()
                        elif "Anons/Blaze" in file_path:
                            voice = "(Game) Sonic_"+splitted[3].title()
                        
                        emotions = splitted[4].lower().split(" ")   # e.g: neutral
                        noise_level = splitted[5].lower()           # e.g: "" = clean, "noisy" = Noisy, "very noisy" = Very Noisy
                        filename_quote = unidecode(splitted[6]) # missing question marks
                    except:
                        if (os.path.basename(dir_).startswith("p")): # if VCTK then use this
                            emotions = []; noise_level = ""; timestamp = "00_00_00"; voice = "VCTK (News Extracts)_"+os.path.basename(dir_)
                        else:
                            print("'"+file_path+"' is not a valid filename")
                            invalid_clip_count+=1; continue
                # ---- VCTK and Clipper Sliced DATASET ----
                try:
                    try:
                        with open(file_path.replace(".wav",".txt"), 'r', encoding="utf-8") as file:
                            txt_quote = unidecode(file.read().replace('\n', '')) # Once upon a time.
                    except:
                        with open(file_path.replace(".wav",".txt"), 'r', encoding="latin-1") as file:
                            txt_quote = unidecode(file.read().replace('\n', '')) # Once upon a time.
                except Exception as ex:
                    print(ex)
                    invalid_clip_count+=1; continue
                if voice.title() in list(metadata.keys()):
                    metadata[str(voice).title()].append({"file_path": file_path, "timestamp": timestamp, "emotions": emotions, "noise_level": noise_level, "quote": txt_quote})
                else:
                    metadata[str(voice).title()] = [{"file_path": file_path, "timestamp": timestamp, "emotions": emotions, "noise_level": noise_level, "quote": txt_quote}]
            else:
                continue
    print(str(total_clip_count)+" Total Clips")
    print(str(short_clip_count)+" clips are too short")
    print(str(long_clip_count)+" clips are too long")
    print(str(invalid_clip_count)+" clips are invalid (bad filename or missing TXT)")
    print(str(total_clip_count-(short_clip_count+long_clip_count+invalid_clip_count))+" clips written into metadata dict")


def write_datasets(speaker_id = 0, permitted_noise_levels = [""], minimum_clips=3, start_token="", end_token="", percentage_training_data=0.96): 
    print("Filtering Metadata for desired files")
    multi_speaker_lines = []
    speaker_ids = []
    speaker_ids_done = []
    for voice in list(metadata.keys()):
        meta = metadata[voice] # meta == [{file_path: "", timestamp: "00_00_05", emotions: ["neutral"], noise_level: "", quote = "Once upon a time."}, .... , ....]
        if len(meta) < minimum_clips: continue # ignore voices with less than 3 clips of audio
        single_speaker_lines = []
        for clip in meta:
            if (clip["noise_level"] in permitted_noise_levels):
                single_speaker_lines.append(clip["file_path"]+"|"+start_token+clip["quote"]+end_token)
                multi_speaker_lines.append (clip["file_path"]+"|"+start_token+clip["quote"]+end_token+"|"+str(speaker_id))
                if speaker_id not in speaker_ids_done: speaker_ids_done.append(speaker_id); speaker_ids.append(f"|{voice}|{speaker_id}")
        speaker_id+=1 # next speaker_id for next voice
    # shuffle stuff
    shuffled_multi_speaker_lines = multi_speaker_lines
    shuffle(shuffled_multi_speaker_lines)
    num_clips = len(shuffled_multi_speaker_lines)
    train_end = int(num_clips * percentage_training_data)
    train_arr = shuffled_multi_speaker_lines[:train_end]; validation_arr = shuffled_multi_speaker_lines[train_end:]
    
    # also make unshuffled stuff (sorted by speaker_id)
    unshuffled_multi_speaker_lines = []
    for i in range(len(list(metadata.keys()))):
        for line in multi_speaker_lines:
            if line.split("|")[2] == str(i): unshuffled_multi_speaker_lines.append(line)
    # Write all this crap to files
    write_files(unshuffled_multi_speaker_lines, speaker_ids, train_arr, validation_arr, root_dir=DIRECTORY_GLOBAL)


def write_files(multi_speaker_lines, speaker_ids, train_arr, val_arr, root_dir):
    output_directory = os.path.join(root_dir,"filelists")
    print(f"Writing Metadata files to {output_directory}.\nPlease Wait.")
    if not os.path.exists(output_directory): os.makedirs(output_directory)
    
    # generate files of speaker ID's
    text_file = open(os.path.join(output_directory,"speaker_ids.txt"), "w", encoding="utf-8")
    text_file.write("\n".join(speaker_ids)); text_file.close()
    
    # generate mel text dataset metadata
    text_file = open(os.path.join(output_directory,"mel_train_taca2.txt"), "w", encoding="utf-8")
    text_file.write("\n".join(train_arr).replace(".wav|",".npy|")); text_file.close()
    arpabet(os.path.join(output_directory,"mel_train_taca2.txt"),os.path.join(output_directory,"mel_train_taca2_arpa.txt"))
    # generate mel arpabet dataset metadata
    text_file = open(os.path.join(output_directory,"mel_validation_taca2.txt"), "w", encoding="utf-8")
    text_file.write("\n".join(val_arr).replace(".wav|",".npy|")); text_file.close()
    arpabet(os.path.join(output_directory,"mel_validation_taca2.txt"),os.path.join(output_directory,"mel_validation_taca2_arpa.txt"))
    # generate mel merged dataset metadata
    concat_text([os.path.join(output_directory,"mel_train_taca2.txt"), os.path.join(output_directory,"mel_train_taca2_arpa.txt")], os.path.join(output_directory,"mel_train_taca2_merged.txt"))
    concat_text([os.path.join(output_directory,"mel_validation_taca2.txt"), os.path.join(output_directory,"mel_validation_taca2_arpa.txt")], os.path.join(output_directory,"mel_validation_taca2_merged.txt"))
    print("mel filepaths ready")
    
    # generate text dataset metadata
    text_file = open(os.path.join(output_directory,"unshuffled_taca2.txt"), "w", encoding="utf-8")
    text_file.write("\n".join(multi_speaker_lines)); text_file.close()
    
    # generate text dataset metadata
    text_file = open(os.path.join(output_directory,"train_taca2.txt"), "w", encoding="utf-8")
    text_file.write("\n".join(train_arr)); text_file.close()
    arpabet(os.path.join(output_directory,"train_taca2.txt"),os.path.join(output_directory,"train_taca2_arpa.txt"))
    # generate arpabet dataset metadata
    text_file = open(os.path.join(output_directory,"validation_taca2.txt"), "w", encoding="utf-8")
    text_file.write("\n".join(val_arr)); text_file.close()
    arpabet(os.path.join(output_directory,"validation_taca2.txt"),os.path.join(output_directory,"validation_taca2_arpa.txt"))
    # generate merged dataset metadata
    concat_text([os.path.join(output_directory,"train_taca2.txt"), os.path.join(output_directory,"train_taca2_arpa.txt")], os.path.join(output_directory,"train_taca2_merged.txt"))
    concat_text([os.path.join(output_directory,"validation_taca2.txt"), os.path.join(output_directory,"validation_taca2_arpa.txt")], os.path.join(output_directory,"validation_taca2_merged.txt"))
    
    print("Finished!")


def convert_dir_to_wav(directory, SAMPLE_RATE=48000, BIT_DEPTH=2, ignore_dirs=["Noise samples"], continue_from=0, skip_existing=False):
    skip = 0
    tqdm.write("Converting flacs to wavs")
    for index, file_path in tqdm(enumerate(glob(directory+"/**/*.flac", recursive=True)), total=len(glob(directory+"/**/*.flac", recursive=True)), smoothing=0): # recursive directory search
        if index < continue_from: continue
        if skip_existing and os.path.exists(file_path.replace(".flac",".wav")): continue
        for filter_dir in ignore_dirs:
            if filter_dir in file_path: tqdm.write("Skipping "+file_path); skip = 1; break
        if skip: skip = 0; continue
        if file_path.endswith("_mic1.flac"):
            os.rename(file_path,file_path.replace("_mic1.flac",".flac"))
        if file_path.endswith("_mic2.flac"): tqdm.write("Skipping "+file_path); continue
        if file_path.endswith(".flac"):
            #tqdm.write(file_path+" --> "+file_path.replace(".flac",".wav"))
            os.system('flac --decode "'+(file_path.replace("_mic1.flac",".flac"))+'" -f -s')


def convert_dir_to_wav_multiprocess(file_paths_arr, SAMPLE_RATE=48000, BIT_DEPTH=2, ignore_dirs=["Noise samples","_Noisy_","_Very Noisy_"], continue_from=0, skip_existing=False):
    skip = 0
    batch_size = 10
    commands = []
    for file_path in tqdm(file_paths_arr, smoothing=0.01): # recursive directory search
        if skip_existing and os.path.exists(file_path.replace(".flac",".wav")): continue
        if any([filter_dir in file_path for filter_dir in ignore_dirs]):
            continue
        if file_path.endswith("_mic1.flac"):
            os.rename(file_path,file_path.replace("_mic1.flac",".flac"))
        if file_path.endswith("_mic2.flac"):
            continue
        if file_path.endswith(".flac"):
            #song = AudioSegment.from_flac(file_path)
            #song.export(file_path.replace(".flac",".wav"), format = "wav", bitrate=str(SAMPLE_RATE))
            commands.append(f'flac --decode "{file_path}" -f -s')
            #commands.append(f'ffmpeg -y -i "{file_path}" -ar {SAMPLE_RATE} -ac 1 "{file_path.replace(".flac",".wav")}" -hide_banner -loglevel warning')
        if len(commands) >= batch_size:
            os.system(";".join(commands))
            commands = []
    if len(commands):
        os.system(";".join(commands))
        commands = []


def set_wavs_to_mono(directory):
    tqdm.write("Setting wavs to mono")
    for file_path in tqdm(glob(directory+"/**/*.wav", recursive=True), smoothing=0): # recursive directory search
        #tqdm.write(file_path)
        sound = AudioSegment.from_wav(file_path)
        if sound.channels > 1:
            sound = sound.set_channels(1)
            sound.export(file_path, format="wav")


def set_wavs_to_mono_multiprocess(file_paths_arr):
    for file_path in file_paths_arr: # recursive directory search
        sound = AudioSegment.from_wav(file_path)
        if sound.channels > 1:
            tqdm.write(file_path)
            sound = sound.set_channels(1)
            sound.export(file_path, format="wav")


def normalize_wav_volumes_mixmode(directory, amplitude=0.08):
    subdirectories = [x[0] for x in os.walk(directory)]
    for subdirectory in subdirectories:
        os.system(f"normalize-audio -w 16 -a {amplitude} -b '{subdirectory}/'*.wav")


def pad_wavs():
    tqdm.write("Padding Wavs")
    os.system("powershell '/media/cookie/Samsung PM961/Pad_Dataset.ps1'")
    tqdm.write("Replacing wavs with Padded wavs")
    force_move_dir("/media/cookie/Samsung 860 QVO/ClipperDatasetV2_Padded","/media/cookie/Samsung 860 QVO/ClipperDatasetV2")


def fix_wavs(directory, SAMPLE_RATE=48000, BIT_DEPTH=2):
    tqdm.write("Setting Wavs to 48Khz 16 bit Mono")
    #os.system("powershell '/media/cookie/Samsung PM961/Fix_Dataset.ps1'")
    reset_directory_structure(directory, directory.replace("/ClipperDatasetV2","/ClipperDatasetV2_Fixed"))
    for file_path in tqdm(glob(directory+"/**/*.wav", recursive=True), smoothing=0): # recursive directory search
        file_pathB = file_path.replace("/ClipperDatasetV2","/ClipperDatasetV2_Fixed")
        os.system('sox "'+file_path+'" -r '+str(SAMPLE_RATE)+' -c 1 -b '+str(BIT_DEPTH*8)+' "'+file_pathB+'"')
    tqdm.write("Replacing wavs with 48Khz 16 bit Mono wavs")
    force_move_dir("/media/cookie/Samsung 860 QVO/ClipperDatasetV2_Fixed","/media/cookie/Samsung 860 QVO/ClipperDatasetV2")


def clean_VCTK(directory, noiseprof_path, denoise_strength=0.40):
    tqdm.write("Cleaning VCTK Wavs")
#    os.system("powershell '/media/cookie/Samsung PM961/RemoveNoise_Dataset.ps1'") # old Powershell method
    reset_directory_structure(directory, directory.replace("/wav","/wavCleaned"))
    for file_path in tqdm(glob(directory+"/**/*.wav", recursive=True), smoothing=0): # recursive directory search
        os.system('sox "'+file_path+'" "'+(file_path.replace("/wav","/wavCleaned"))+'" noisered "'+noiseprof_path+f'" {denoise_strength}')
    tqdm.write("Replacing wavs with Cleaned wavs")


def clean_VCTK_multiprocess(file_paths_arr, noiseprof_path, denoise_strength=0.40):
    tqdm.write("Cleaning VCTK Wavs")
#    os.system("powershell '/media/cookie/Samsung PM961/RemoveNoise_Dataset.ps1'") # old Powershell method
    for file_path in tqdm(file_paths_arr, smoothing=0.0): # recursive directory search
        os.system('sox "'+file_path+'" "'+(file_path.replace("/wav","/wavCleaned"))+'" noisered "'+noiseprof_path+f'" {denoise_strength}')
    tqdm.write("Replacing wavs with Cleaned wavs")


def AIO_wavs(directory, PADDING_BEFORE_CLIP=0.025, PADDING_AFTER_CLIP=0.025, SAMPLE_RATE=48000, BIT_DEPTH=2):
    tqdm.write("Normalizing, Padding and Fixing Dataset")
    #os.system("powershell '/media/cookie/Samsung PM961/All_in_one_Dataset.ps1'")

    for file_path in tqdm(glob(directory+"/**/*.wav", recursive=True), smoothing=0): # recursive directory search
        file_pathB = file_path.replace("/ClipperDatasetV2","/ClipperDatasetV2_Padded")
        #os.system('normalize-audio "'+file_path+'" -a 0.03125; sox "'+file_path+'" -r '+str(SAMPLE_RATE)+' -c 1 -b '+str(BIT_DEPTH*8)+' "'+file_pathB+'"; sox "'+file_pathB+'" "'+file_path+'" pad '+str(PADDING_BEFORE_CLIP)+' '+str(PADDING_BEFORE_CLIP))
        os.system('sox "'+file_path+'" -r '+str(SAMPLE_RATE)+' -c 1 -b '+str(BIT_DEPTH*8)+' "'+file_pathB+'"; sox "'+file_pathB+'" "'+file_path+'" pad '+str(PADDING_BEFORE_CLIP)+' '+str(PADDING_BEFORE_CLIP))


def AIO_wavs_multiprocess(file_paths_arr, PADDING_BEFORE_CLIP=0.0, PADDING_AFTER_CLIP=0.0125, SAMPLE_RATE=48000, BIT_DEPTH=2):
    commands = []
    batch_size = 4
    for file_path in tqdm(file_paths_arr, smoothing=0.0): # recursive directory search
        file_pathB = file_path.replace("/ClipperDatasetV2","/ClipperDatasetV2_Padded")
        assert file_path != file_pathB
        #os.system('normalize-audio "'+file_path+'" -a 0.03125; sox "'+file_path+'" -r '+str(SAMPLE_RATE)+' -c 1 -b '+str(BIT_DEPTH*8)+' "'+file_pathB+'"; sox "'+file_pathB+'" "'+file_path+'" pad '+str(PADDING_BEFORE_CLIP)+' '+str(PADDING_BEFORE_CLIP))
        #os.system('sox "'+file_path+'" -r '+str(SAMPLE_RATE)+' -c 1 -b '+str(BIT_DEPTH*8)+' "'+file_pathB+'"; sox "'+file_pathB+'" "'+file_path+'" pad '+str(PADDING_BEFORE_CLIP)+' '+str(PADDING_BEFORE_CLIP))
        if PADDING_BEFORE_CLIP and PADDING_AFTER_CLIP:
            commands.append('sox "'+file_path+'" -r '+str(SAMPLE_RATE)+' -c 1 -b '+str(BIT_DEPTH*8)+' "'+file_pathB+'"; sox "'+file_pathB+'" "'+file_path+'" pad '+str(PADDING_BEFORE_CLIP)+' '+str(PADDING_BEFORE_CLIP))
        else:
            commands.append('sox "'+file_path+'" -r '+str(SAMPLE_RATE)+' -c 1 -b '+str(BIT_DEPTH*8)+' "'+file_pathB+'"; mv -f "'+file_pathB+'" "'+file_path+'"')
        if len(commands) >= batch_size:
            os.system(";".join(commands))
            commands = []
    if len(commands):
        os.system(";".join(commands))
        commands = []


def trim_wavs(directory, SAMPLE_RATE=48000, top_db=30, window_length=8192, hop_length=128, ref=np.max):
    tqdm.write("Trimming Wavs")
    for file_path in tqdm(glob(directory+"/**/*.wav", recursive=True), smoothing=0): # recursive directory search
        try:
            sound, _ = librosa.core.load(file_path, sr=SAMPLE_RATE)
            trimmed_sound, index = librosa.effects.trim(sound, top_db=top_db, frame_length=window_length, hop_length=hop_length, ref=ref) # gonna be a little messed up for different sampling rates
        except Exception as ex:
            tqdm.write("\n\n"+file_path+" is corrupt"+str(ex)); os.system('rm "'+file_path+'"')
        librosa.output.write_wav(file_path, trimmed_sound, SAMPLE_RATE)


def trim_wavs_multiprocess(file_paths_arr, SAMPLE_RATE=48000, margin_left=0, margin_right=0, top_db=30, window_length=8192, hop_length=128, ref=np.max, preemphasis_strength=0.50):
    
    for file_path in tqdm(file_paths_arr, smoothing=0.0): # recursive directory search
        try:
            sound, _ = librosa.core.load(file_path, sr=SAMPLE_RATE, mono=True) # resample to SAMPLE_RATE and downmix to mono. This actually works!
            trimmed_sound = sound
            
            # run each trim pass
            for margin_left_, margin_right_, top_db_, window_length_, hop_length_, ref_, preemphasis_strength_ in zip(margin_left, margin_right, top_db, window_length, hop_length, ref, preemphasis_strength):
                if preemphasis_strength_:
                    sound_filt = librosa.effects.preemphasis(trimmed_sound, coef=preemphasis_strength_)
                    _, index = librosa.effects.trim(sound_filt, top_db=top_db_, frame_length=window_length_, hop_length=hop_length_, ref=ref_) # gonna be a little messed up for different sampling rates
                else:
                    _, index = librosa.effects.trim(trimmed_sound, top_db=top_db_, frame_length=window_length_, hop_length=hop_length_, ref=ref_) # gonna be a little messed up for different sampling rates
                trimmed_sound = trimmed_sound[index[0]-margin_left_:index[1]+margin_right_]
            
            print(index[0], index[1], len(sound)-index[1], len(trimmed_sound) == index[1]-index[0])
            
            if len(sound) != len(trimmed_sound): # assuming the audio file has changed length, write the new file
                librosa.output.write_wav(file_path, trimmed_sound, SAMPLE_RATE)
                tqdm.write(f"TRIMMING TRIMMING DO TRIMMING")
            else:
                tqdm.write(f"NOT TRIM")
        except Exception as ex:
            tqdm.write("\n"+file_path+" failed\n"+str(ex)); os.system('rm "'+file_path+'"')


def process_wavs_multiprocess(file_paths_arr, filt_type, filt_cutoff_freq, filt_order,
        trim_margin_left, trim_margin_right, trim_top_db, trim_window_length, trim_hop_length, trim_ref, trim_preemphasis_strength,
        SAMPLE_RATE=48000,
        MIN_SAMPLE_RATE=22050,
        BIT_DEPTH=2,
        ignore_dirs=["Noise samples","_Noisy_","_Very Noisy_"],
        skip_existing=False,
        ):
    import soundfile as sf
    import scipy
    from scipy import signal
    
    skip = 0
    prev_sr = 0
    for file_path in tqdm(file_paths_arr, smoothing=0.0): # recursive directory search
        out_path = file_path.replace(".flac",".wav")
        if skip_existing and os.path.exists(out_path):
            continue
        if any([filter_dir in file_path for filter_dir in ignore_dirs]):
            continue
        if file_path.endswith("_mic1.flac"):
            os.rename(file_path, file_path.replace("_mic1.flac",".flac"))
        if file_path.endswith("_mic2.flac"):
            continue
        
        native_sound, native_SR = sf.read(file_path, always_2d=True)
        native_sound = native_sound[:,0]# take first channel (either mono or left typically)
        native_sound = np.asfortranarray(native_sound).astype('float64') # and ensure the audio is contiguous
        
        if native_SR < MIN_SAMPLE_RATE: # skip any files with native_SR below the minimum
            continue
        if native_SR != SAMPLE_RATE: # ensure all audio is same Sample Rate
            try:
                sound = librosa.core.resample(native_sound, native_SR, SAMPLE_RATE)
            except ValueError as ex:
                print(ex)
                print(file_path)
                print(native_SR)
                print(len(native_sound))
                raise ValueError(ex)
        else:
            sound = native_sound
        
        if max(np.amax(native_sound), -np.amin(native_sound)) > (2**23): # if samples exceed values possible at 24 bit
            sound = (sound / 2**(31-15))#.astype('int16') # change bit depth from 32 bit to 16 bit
        elif max(np.amax(native_sound), -np.amin(native_sound)) > (2**15): # if samples exceed values possible at 16 bit
            sound = (sound / 2**(23-15))#.astype('int16') # change bit depth from 24 bit to 16 bit
        
        # apply audio filters
        for type_, freq_, order_ in zip(filt_type, filt_cutoff_freq, filt_order): # eg[ ['lp'], [40], [10] ] # i.e [type, freq, strength]
            sos = signal.butter(order_, freq_, type_, fs=SAMPLE_RATE, output='sos') # calcuate filter somethings
            sound = signal.sosfilt(sos, sound) # apply filter
        
        original_len = len(sound)
        # apply audio trimming
        for i, (margin_left_, margin_right_, top_db_, window_length_, hop_length_, ref_, preemphasis_strength_) in enumerate(zip(trim_margin_left, trim_margin_right, trim_top_db, trim_window_length, trim_hop_length, trim_ref, trim_preemphasis_strength)):
            if preemphasis_strength_:
                sound_filt = librosa.effects.preemphasis(sound, coef=preemphasis_strength_)
                _, index = librosa.effects.trim(sound_filt, top_db=top_db_, frame_length=window_length_, hop_length=hop_length_, ref=ref_) # gonna be a little messed up for different sampling rates
            else:
                _, index = librosa.effects.trim(sound, top_db=top_db_, frame_length=window_length_, hop_length=hop_length_, ref=ref_) # gonna be a little messed up for different sampling rates
            sound = sound[max(index[0]-margin_left_, 0):index[1]+margin_right_]
            assert len(sound), f"Audio trimmed to 0 length by pass {i+1}\nconfig = {[margin_left_, margin_right_, top_db_, window_length_, hop_length_, ref_]}\nFile_Path = '{file_path}'"
        
        sf.write(out_path, sound, SAMPLE_RATE)


def multi_filter_wavs_multiprocess(file_paths_arr, type, cutoff_freq, order, SAMPLE_RATE=48000):
    import soundfile as sf
    import scipy
    from scipy import signal
    
    prev_sr = 0
    for file_path in tqdm(file_paths_arr, smoothing=0.0): # recursive directory search
            audio, _ = librosa.core.load(file_path, sr=SAMPLE_RATE)
            
            for type_, freq_, order_ in zip(type, cutoff_freq, order): # eg[ ['lp'], [40], [10] ] # i.e [type, freq, strength]
                sos = signal.butter(order_, freq_, type_, fs=SAMPLE_RATE, output='sos') # calcuate filter somethings
                audio = signal.sosfilt(sos, audio) # apply filter
            
            librosa.output.write_wav(file_path, audio, SAMPLE_RATE)
            #sf.write(file_path, audio, sample_rate) # write back to disk


def high_pass_filter_wavs_multiprocess(file_paths_arr, cutoff_freq, strength):
    import soundfile as sf
    import scipy
    from scipy import signal
    
    prev_sr = 0
    for file_path in tqdm(file_paths_arr, smoothing=0.0): # recursive directory search
            audio, sample_rate = sf.read(file_path) # load audio to RAM
            if sample_rate != prev_sr:
                sos = signal.butter(strength, cutoff_freq, 'hp', fs=sample_rate, output='sos') # calcuate filter somethings
                prev_sr = sample_rate
            filtered_audio = signal.sosfilt(sos, audio) # apply filter
            sf.write(file_path, filtered_audio, sample_rate) # write back to disk


def low_pass_filter_wavs_multiprocess(file_paths_arr, cutoff_freq, strength):
    import soundfile as sf
    import scipy
    from scipy import signal
    
    prev_sr = 0
    for file_path in tqdm(file_paths_arr, smoothing=0.0): # recursive directory search
            audio, sample_rate = sf.read(file_path) # load audio to RAM
            if cutoff_freq*2 < sample_rate:
                if sample_rate != prev_sr:
                        sos = signal.butter(strength, cutoff_freq, 'lp', fs=sample_rate, output='sos') # calcuate filter somethings
                        prev_sr = sample_rate
                filtered_audio = signal.sosfilt(sos, audio) # apply filter
                sf.write(file_path, filtered_audio, sample_rate) # write back to disk


def reset_padding(directory, SAMPLE_RATE=48000):
    trim_wavs(directory, SAMPLE_RATE=SAMPLE_RATE, top_db=60, window_length=256)


def multithread_directory_wavs(function, directory, threads=16):
    from random import shuffle
    p = Pool(threads)
    file_paths = glob(directory+"/**/*.wav", recursive=True)
    shuffle(file_paths)
    split_file_paths = list(even_split(file_paths,threads))
    print(p.map(function, split_file_paths))


def multiprocess_directory_wavs(function, directory, threads=16):
    from random import shuffle
    p = Pool(threads)
    file_paths = glob(directory+"/**/*.wav", recursive=True)
    shuffle(file_paths)
    split_file_paths = list(even_split(file_paths,threads))
    #with concurrent.futures.ProcessPoolExecutor() as executor:
    #    print(executor.map(function, split_file_paths))
    print(p.map(function, split_file_paths))


def multiprocess_directory_flacs(function, directory, threads=16):
    from random import shuffle
    p = Pool(threads)
    file_paths = glob(directory+"/**/*.flac", recursive=True)
    shuffle(file_paths)
    split_file_paths = list(even_split(file_paths,threads))
    #with concurrent.futures.ProcessPoolExecutor() as executor:
    #    print(executor.map(function, split_file_paths))
    print(p.map(function, split_file_paths))
    
    
def multiprocess_filearray(function, file_paths, threads=16):
    p = Pool(threads)
    split_file_paths = list(even_split(file_paths,threads))
    #with concurrent.futures.ProcessPoolExecutor() as executor:
    #    print(executor.map(function, split_file_paths))
    print(p.map(function, split_file_paths))


def convertWavsToFlac(array, SAMPLE_RATE=48000):
    from pydub import AudioSegment
    for path in array:
        if path.endswith('.wav'):
            song = AudioSegment.from_wav(path)
            song.export(path.replace(".wav",".flac"),format = "flac", bitrate=str(SAMPLE_RATE))


def NancySplitRawIntoClips(directory, label_folder, Nancy_CorpusToArchive, SAMPLE_RATE=96000):
    """
    Take original 96Khz studio files and match them with the filenames used for the quote files.
    """
    Nancy_ignore = {studio: output for output, studio, exception in [line.split("\t") for line in ((open(Nancy_CorpusToArchive, "r").read()).splitlines())] if exception}
    print(Nancy_ignore)
    Nancy_lookup = {studio: output for output, studio, exception in [line.split("\t") for line in ((open(Nancy_CorpusToArchive, "r").read()).splitlines())]}
    
    os.makedirs(os.path.join(directory,'Sliced'), exist_ok=True)
    available_labels = ["/".join(x.split("/")[-1:]) for x in glob(label_folder+"/*.txt")]
    available_audio = glob(directory+"/*.wav")
    for audio_file in available_audio:
        print(audio_file)
        audio_filename = "/".join(audio_file.split("/")[-1:])
        audio_basename = audio_filename.replace(".wav","")
        audio_basename = audio_basename.replace("341_763","343_763") # exception, easier than rewriting entire label file
        ID_offset = int(audio_basename.split("_")[-2]) - 1
        ID_end = int(audio_basename.split("_")[-1]) - 1
        Prepend_ID = "_".join(audio_basename.split("_")[:-2]) # empty unless ARCTIC or LTI files
        if Prepend_ID: Prepend_ID += "_"
        if audio_filename.replace(".wav",".txt") in available_labels:
            label_path = os.path.join(label_folder, audio_filename.replace(".wav",".txt") ) # get label file
            beeps = []
            for line in ((open(label_path, "r").read()).splitlines()):
                beeps+=[line.split("\t")] # [beep_start, beep_stop, ID]
            print("beep count", len(beeps))
            print("ID_offset", ID_offset)
            print("ID_end", ID_end)
            print("end - offset", ID_end-ID_offset)
            assert (len(beeps)-1) == (ID_end-ID_offset), "Ensure each beep is labelled and matches the ArchiveMap"
            sound, _ = librosa.core.load(audio_file, sr=SAMPLE_RATE)
            for i in range(len(beeps)):
                clip_start = int(float(beeps[i][1])*SAMPLE_RATE) # end of previous beep
                clip_end = int(float(beeps[i+1][0])*SAMPLE_RATE) if i+1 < len(beeps) else len(sound) # start of next beep or end of file if no beeps left
                ID = Prepend_ID + str(ID_offset + int(beeps[i][2]))
                if ID in Nancy_ignore.keys(): continue
                print(ID,"-->",Nancy_lookup[ID])
                ID = Nancy_lookup[ID]
                clip_outpath = os.path.join(directory, 'Sliced', ID+".wav")
                sound_clipped = sound[clip_start:clip_end]
                librosa.output.write_wav(clip_outpath, sound_clipped, SAMPLE_RATE)
        else:
            print(audio_file, "doesn't have an available label")


def uniqueElems(mylist):
    used = set()
    return [x for x in mylist if x not in used and (used.add(x) or True)]


def ClipperSplitRawIntoClips(working_dir, source_dir, label_dir, verbose=False, explicit_merge=False, MAX_FILENAME_LENGTH=120):
    """
    Take original MLP Episode files and split them into individual clips.
    Also merge neighbouring clips to create longer clips and merge clips of ponies speaking to each-other.
    
    S1-S8 Episode Audio: https://files.catbox.moe/2q4p2z.torrent
    Clippers MEGA Folder: https://mega.nz/#F!L952DI4Q!nibaVrvxbwgCgXMlPHVnVw!fwYlhK7B
    
    INPUTS:
        working_dir - will contain the outputs and any temp files
        source_dir - should contain original unclipped episode files downloaded from https://files.catbox.moe/2q4p2z.torrent
        label_dir - should contain episode labels from clippers MEGA folder
    RETURNS:
        null
    """
    os.makedirs(os.path.join(working_dir,'Sliced'), exist_ok=True)
    label_paths = glob(label_dir+"/*.txt")
    unclipped_paths = glob(source_dir+"/**/*.wav", recursive=True)
    
    # ---------- LINK THE LABELS TO FILES ----------
    linked_paths = [] # dictionary of labels -> unclipped wavs
    for label_path_ in label_paths:
        label_path = label_path_ # get a temp varable to fuck about with
        if any(True for x in ["_izo.txt","_original.txt","_unmix.txt"] if x in label_path): continue # ignore any izo, orignal or unmix labels.
        if "fim_s01" in label_path:
            episode = label_path.split("fim_s01")[1][:3]
            #print(episode)
            matching_audio = [x for x in unclipped_paths if ("mlp.s01"+episode in x)] # get all audio files that potentially match this label file.
            if len(matching_audio) > 1: print("Too many potential matches:\nmaching_audio = ", matching_audio); raise Exception("Multiple episodes match a label file.")
            if not len(matching_audio): print("label_path_ = ", label_path_); raise Exception("Label has no matching audio file.")
            
            linked_paths.append({label_path_: matching_audio[0]})
            unclipped_paths.remove(matching_audio[0])
            continue
        elif "_outtakes.txt" in label_path:
            continue # for later
        elif "_special source.txt" in label_path:
            continue # for later
        elif "fim_s09" in label_path:
            continue # for later
        elif "fim_s" in label_path: # all seasons other than s01
            episode = label_path.split("fim_s")[1][2:5]
            season = "S"+label_path.split("fim_s")[1][:2]
            s_ep = (season + episode).upper()
            
            matching_audio = [x for x in unclipped_paths if (s_ep in x)] # get all audio files that potentially match this label file.
            if len(matching_audio) > 1: print("Too many potential matches:\nmaching_audio = ", matching_audio); raise Exception("Multiple episodes match a label file.")
            if not len(matching_audio): print("label_path_ = ", label_path_); raise Exception("Label has no matching audio file.")
            linked_paths.append({label_path_: matching_audio[0]})
            unclipped_paths.remove(matching_audio[0])
            continue
    
    if verbose:
        for l in linked_paths:
            print(l, "")
    
    for i, linked_path in enumerate(linked_paths):
        label_path, audio_path = list(linked_path.items())[0]
        base_audio_dirs = []
        labels = []
        for line in ((open(label_path, "r").read()).splitlines()):
            labels+=[line.split("\t")] # [label_start, label_stop, time_voice_emotion_noise_quote]
        
        audio, _ = librosa.core.load(audio_path, sr=SAMPLE_RATE)
        
        # relative dir to store audio data
        if "fim_s" in label_path: # all seasons other than s01
            episode = label_path.split("fim_s")[1][2:5]
            season = "S"+label_path.split("fim_s")[1][:2]
            s_ep = (season + episode).upper()
            base_audio_dirs.append(season)
            base_audio_dirs.append(episode)
        
        prev_label = [-10, -10, "00_00_00_Null_Null_Noisy_"]
        for label in tqdm(labels, leave=False, desc=f"{i}/{len(linked_paths)}"):
            audio_dirs = base_audio_dirs # get relative output path back to ["S01","E01"]
            label_start, label_stop, info = label
            *time_stamp, voice, str_emotion, noise_level, quote = info.split("_") # 00_00_05_Celestia_Neutral__Once upon a time. -> [["00","00","05"],"Celestia","Neutral","","Once upon a time."]
            emotions = str_emotion.split(" ") # "Happy Shouting" -> ["Happy","Shouting"]
            
            prev_label_start, prev_label_stop, prev_info = prev_label
            *prev_time_stamp, prev_voice, prev_str_emotion, prev_noise_level, prev_quote = prev_info.split("_") # 00_00_05_Celestia_Neutral__Once upon a time. -> [["00","00","05"],"Celestia","Neutral","","Once upon a time."]
            prev_emotions = prev_str_emotion.split(" ") # "Happy Shouting" -> ["Happy","Shouting"]
            
            time_between_clips = float(label_start) - float(prev_label_stop) # time between start of this clip and end of the last clip
            prev_label_start_sample = int(float(prev_label_start)*SAMPLE_RATE)
            prev_label_stop_sample = int(float(prev_label_stop)*SAMPLE_RATE)
            label_start_sample = int(float(label_start)*SAMPLE_RATE)
            label_stop_sample = int(float(label_stop)*SAMPLE_RATE)
            
            # get more accurate silent time between clips
            if (float(prev_label_stop)-float(prev_label_start)) > 0.025: # if time_between clips longer than 25ms and previous clip has duration longer than 25ms
                def return_ref(input):
                    return 0.0000250# 0.0000020 for trimming silence but leaving breathing in. # 0.0000250 for trimming silence and also quiet breathing.
                clipped_audio = audio[prev_label_start_sample:prev_label_stop_sample] # previous sample
                index = librosa.effects.trim(clipped_audio, top_db=0, frame_length=int(SAMPLE_RATE*0.100), hop_length=int(SAMPLE_RATE*0.0125), ref=return_ref)[1] # gonna be a little messed up for different sampling rates
                prev_label_stop_speech_sample = prev_label_start_sample+index[1] # when previous loud speech stop, ignores breating and is a rough trim
                #prev_label_start_sample = prev_label_start_sample+index[0]
                clipped_audio = audio[label_start_sample:label_stop_sample] # current sample
                index = librosa.effects.trim(clipped_audio, top_db=0, frame_length=int(SAMPLE_RATE*0.100), hop_length=int(SAMPLE_RATE*0.0125), ref=return_ref)[1] # gonna be a little messed up for different sampling rates
                #label_stop_sample = label_start_sample+index[1]
                label_start_speech_sample = label_start_sample+index[0] # when loud speech starts, ignores breating and is a rough trim
                time_between_clips = (label_start_speech_sample - prev_label_stop_speech_sample)/SAMPLE_RATE # time where there is only silence or very quiet breathing.
                
                # this block does the same as above, just in one dense line and gets the silent time rather than quiet time between clips.
                def return_ref(input):
                    return 0.0000020# 0.0000020 for trimming silence but leaving breathing in. # 0.0000250 for trimming silence and also quiet breathing.
                silent_time_between_clips = ((label_start_sample+librosa.effects.trim(audio[label_start_sample:label_stop_sample], top_db=0, frame_length=int(SAMPLE_RATE*0.100), hop_length=int(SAMPLE_RATE*0.0125), ref=return_ref)[1][0]) - (prev_label_start_sample+librosa.effects.trim(audio[prev_label_start_sample:prev_label_stop_sample], top_db=0, frame_length=int(SAMPLE_RATE*0.100), hop_length=int(SAMPLE_RATE*0.0125), ref=return_ref)[1][1]))/SAMPLE_RATE # time where there is only silence or very quiet breathing. 
                
                # get rough volume between clips
                audio_between_clips = audio[prev_label_stop_speech_sample:label_start_speech_sample]
                std_between_clips = np.std(audio_between_clips) if len(audio_between_clips) > 10 else 0
            
            # --- DO STUFF ---
            
            if time_between_clips < 0.8 and noise_level == "" and prev_noise_level == "":
                clipped_audio = audio[prev_label_start_sample:label_stop_sample]
                assert len(clipped_audio), f"LABEL: {label}\nFILE: {label_path}\ndid not process correctly. Output audio has 0 length."
                
                if prev_voice.lower() == voice.lower():
                    audio_dirs = ["Merged Same Speaker"] + audio_dirs
                else:
                    audio_dirs = ["Merged Multi Speaker"] + audio_dirs                    
                
                # cleanup the break point between clips
                if explicit_merge:
                    merged_quote = prev_quote+'#'+quote
                else:
                    if time_between_clips < 0.08:
                        prev_quote = prev_quote[:-1]+"," if prev_quote[-1] == "." else prev_quote
                        merged_quote = prev_quote+' '+quote
                    elif time_between_clips < 0.3:
                        merged_quote = prev_quote+' '+quote
                    elif silent_time_between_clips > 0.4:
                        prev_quote = prev_quote+".." if prev_quote[-1] == "." else prev_quote # "Sentence. Yes?" -> "Sentence... Yes?" if pause has no breathing and last clip ends with "."
                        merged_quote = prev_quote+' '+quote
                    else:
                        merged_quote = prev_quote+' '+quote
                
                # ensure merged quote isn't too long for saving.
                while len(merged_quote) > MAX_FILENAME_LENGTH:
                    merged_quote = " ".join(merged_quote.split(" ")[:-1])+merged_quote[-1] # cut of the last word
                
                filename = f"{'_'.join(prev_time_stamp)}_{' '.join(uniqueElems([prev_voice, voice]))}_{' '.join(uniqueElems(prev_emotions+emotions))}_{noise_level}_{merged_quote}.wav"
                output_folder = os.path.join(working_dir, 'Sliced', *audio_dirs)
                os.makedirs(output_folder, exist_ok=True)
                output_path = os.path.join(output_folder, filename)
                librosa.output.write_wav(output_path, clipped_audio, SAMPLE_RATE)
            
            if False: # normal sliced dialogue
                clipped_audio = audio[label_start_sample:label_stop_sample]
                assert len(clipped_audio), f"LABEL: {label}\nFILE: {label_path}\ndid not process correctly. Output audio has 0 length."
                
                audio_dirs = ["Sliced"] + audio_dirs
                
                filename = f"{'_'.join(prev_time_stamp)}_{' '.join(uniqueElems([prev_voice, voice]))}_{' '.join(uniqueElems(prev_emotions+emotions))}_{noise_level}_{quote}.wav"
                output_folder = os.path.join(working_dir, 'Sliced', *audio_dirs)
                os.makedirs(output_folder, exist_ok=True)
                output_path = os.path.join(output_folder, filename)
                librosa.output.write_wav(output_path, clipped_audio, SAMPLE_RATE)
            
            # --- DO STUFF ---
            prev_label = label


def test_audio_files(directory):
    """
    For different audio files, test whether they trim correctly.
    """
    paths = glob(directory+"/*.wav")
    for audio_path in paths:
        audio, _ = librosa.core.load(audio_path, sr=SAMPLE_RATE)
        def return_ref(input):
            return 0.0000250 # 0.0000020 for trimming silence but leaving breathing in.
                             # 0.0000250 for trimming silence and also quiet breathing.
        index = librosa.effects.trim(audio, top_db=0, frame_length=int(SAMPLE_RATE*0.100), hop_length=int(SAMPLE_RATE*0.0125), ref=return_ref)[1]   
        print("file:", os.path.basename(audio_path),"\t", round(index[0]/SAMPLE_RATE,2), "\t", round((len(audio)-index[1])/SAMPLE_RATE,2), "\tduration:", round((index[1]-index[0])/SAMPLE_RATE,2), "\toriginal_duration:\t", round(len(audio)/SAMPLE_RATE,2), "\tPercent:",round((index[1]-index[0])/len(audio),3))


# ---------Config------------------
dictionary_path = r"/media/cookie/Samsung PM961/TwiBot/tacotron2/filelists/merged_dict"
DIRECTORY_GLOBAL = r"/media/cookie/Samsung 860 QVO/ClipperDatasetV2"
PERCENT_TRAIN = 0.99
PADDING_BEFORE_CLIP= 0.0000 # 0 samples @ 48Khz
PADDING_AFTER_CLIP = 0.0000 # 600 samples @ 48Khz
LEFT_MARGIN = 2400 # extra area before trim triggers
RIGHT_MARGIN = 2400 # extra area after trim triggers
MIN_DURATION = 0.6125 # measured after padding
MAX_DURATION = 16.000 # measured after padding
MIN_CLIPS = 10 # how many clips each speaker must have to be included
PATH_IGNORE = ["Noise samples","_Very Noisy_","_Noisy_","22.05 kHz"]
START_TOKEN=""#"☺"
STOP_TOKEN =""#"␤"
THREADS = 16 # Thread count for multithreaded sections. Normally SSD/HDD bottlenecked anyway.

# Global

# Clippers MLP SlicedDialogue
SlicedDialogue_directory = r"/media/cookie/Samsung 860 QVO/ClipperDatasetV2/SlicedDialogue"

# VCTK
VCTK_directory = r"/media/cookie/Samsung 860 QVO/ClipperDatasetV2/VCTK-Corpus-0.92/wav"
#VCTK_directory = r"/media/cookie/Samsung 860 QVO/ClipperDatasetV2/wav"

VCTK_DENOISE_STRENGTH = 0#0.30
VCTK_Noise_Profile = "/media/cookie/Samsung 860 QVO/ClipperDatasetV2/VCTK-Corpus-0.92/noise.prof"

# Nancy
Nancy_directory = r"/media/cookie/Samsung 860 QVO/ClipperDatasetV2/Blizzard2011"
Nancy_labels = r"/media/cookie/Samsung 860 QVO/ClipperDatasetV2/Blizzard2011_Labels"
Nancy_CorpusToArchive = r"/media/cookie/Samsung 860 QVO/ClipperDatasetV2/Blizzard2011_Labels/NancyCorpusToArchiveMap.txt"
Nancy_Prompts = r"/media/cookie/Samsung 860 QVO/ClipperDatasetV2/Blizzard2011_Labels/prompts.data"

# Clipper unClipper Source Files
clipper_working_dir = r"/media/cookie/MoreStable/SV2TTS_Dataset/Datasets_In_Progress/MLP_Unclipped/MLP_Dialog" # working_dir
clipper_source_dir = r"/media/cookie/MoreStable/SV2TTS_Dataset/Datasets_In_Progress/MLP_Unclipped/MLP_Dialog/Sources_updated_names" # source_dir
clipper_label_dir = r"/media/cookie/MoreStable/SV2TTS_Dataset/Datasets_In_Progress/MLP_Unclipped/MLP_Dialog/Label files" # label_dir

# ---------Less common Config------
skip_already_existing_files = 0 # when converting FLACs to WAVs.
AUDIO_DEPTH = 2 # target Audio, eg. 16 bit audio = 2 (bytes)
SAMPLE_RATE = 48000

# ---------Sox commands------------
#sox noise-audio.wav -n noiseprof noise.prof # generate noise profile
#sox speech.wav cleaned.wav noisered noise.prof 0.5 # clean audio using the noise profile, 0.5 = noise filter strength (between 0.0 and 1.0)
#sox long.mp3 short.mp3 trim 2 -1 # trim by set amounts, (start clip 2 seconds in, end clip 1 second before the original end)
#sox "$path" "$outpath" pad 0 0.1 # add 0 seconds to start, 0.1 seconds to end of clip.
#sox "$path" "$outpath" silence 1 0.01 1.5% reverse silence 1 0.01 1.5% reverse # simple trim silence, recommend using python implementation for better control
#sox "$path" -r 48000 -c 1 -b 16 "$outpath" # convert file to different channels, sample rate and bit-depth.

# ---------Main block--------------
if __name__ == '__main__':
    metadata = {}
    thisdict = get_arpadict(dictionary_path)
    
# ---------Testing/Toying ---------
    
    #DIRECTORY_GLOBAL = r"/media/cookie/Samsung 860 QVO/ClipperDatasetV2/SlicedDialogue/Other/Dr. Who"
    
# ---------Clipper Raw -> Clips && Raw -> Dialogue ---------
    if 0:
        print('Converting source unclipped FLAC files to WAV')
        def func(array):
            convert_dir_to_wav_multiprocess(array, SAMPLE_RATE=SAMPLE_RATE, BIT_DEPTH=AUDIO_DEPTH, skip_existing=skip_already_existing_files)
        multiprocess_directory_flacs(func, clipper_source_dir, threads=4) # multiprocess set wavs to mono
    
    #test_audio_files(clipper_working_dir)
    
    if 0:
        print('Splitting unclipped WAVs to Sliced Dialogue. Merged Clips and Clips with multiple speakers are included.')
        ClipperSplitRawIntoClips(
            clipper_working_dir, # working_dir
            clipper_source_dir, # source_dir
            clipper_label_dir, # label_dir
            )
    
# ---------Nancy Pre-prep (From original WAV Studio Recordings) ---------
    if 0:
        print('Converting Nanacy Original Studio wavs to FLAC.')
        def func(array):
            convertWavsToFlac(array, SAMPLE_RATE=SAMPLE_RATE)
        multiprocess_directory_wavs(func, Nancy_directory, threads=4) # multiprocess set wavs to mono
        
# ---------Nancy Prep (from FLAC Studio Recordings) ---------
    if 0:
        print('Converting source FLAC files to WAV')
        def func(array):
            convert_dir_to_wav_multiprocess(array, SAMPLE_RATE=SAMPLE_RATE, BIT_DEPTH=AUDIO_DEPTH, skip_existing=skip_already_existing_files)
        multiprocess_directory_flacs(func, Nancy_directory) # multiprocess set wavs to mono
        
        print('Clipping Nancy WAV sources into individual chunks.')
        NancySplitRawIntoClips(Nancy_directory, Nancy_labels, Nancy_CorpusToArchive)
        
# --------Convert Nancy Clipped WAV's to FLAC (ready for the normally processing) --------
    if 0:
        print('Converting Nanacy Clipped wavs to FLAC.')
        def func(array):
            convertWavsToFlac(array, SAMPLE_RATE=SAMPLE_RATE)
        multiprocess_directory_wavs(func, os.path.join(Nancy_directory, "Sliced"), threads=4) # multiprocess set wavs to mono
        
# ---------Preprocess wavs---------
    if 0: # convert flac to wav (and optional denoise)
        print('Converting source FLAC files to WAV')
        def func(array):
            convert_dir_to_wav_multiprocess(array, SAMPLE_RATE=SAMPLE_RATE, BIT_DEPTH=AUDIO_DEPTH, ignore_dirs=PATH_IGNORE, skip_existing=skip_already_existing_files)
        multiprocess_directory_flacs(func, DIRECTORY_GLOBAL, threads=THREADS*2) # multiprocess set wavs to mono
        
        if VCTK_DENOISE_STRENGTH:
            print('Denoising VCTK')
            def func(array):
                clean_VCTK_multiprocess(array, VCTK_Noise_Profile, denoise_strength=VCTK_DENOISE_STRENGTH)    
            reset_directory_structure(VCTK_directory, VCTK_directory.replace('/wav',"/wavCleaned"))
            multiprocess_directory_wavs(func, VCTK_directory, threads=THREADS)
            force_move_dir(VCTK_directory.replace("/wav","/wavCleaned"), VCTK_directory)
    
    if 0: # lp/hp filter and trim global
        #        Pass |  1  |  2  |  3  |
        type        = ['hp' ,'hp' ,'lp' ]
        cutoff_freq = [150  ,40   ,18000]
        order       = [4    ,9    ,9    ]
        ref_db   =[46]*5
        win_len  =[9600,4800,2400,1200,600]
        hop_len  =[1200,600 ,300 ,150 ,150]
        ref_f    =[np.amax]*5
        empth    =[0.0]*5
        margin_l =[0]*5
        margin_r =[0]*5
        print("Filtering and Trimming Global WAVs")
        for i in range(len(type)):
            print(f"Filter Pass {i+1}, Global {cutoff_freq[i]}Hz {'highpass' if type[i]=='hp' else 'lowpass'} filter.")
        for i in range(len(win_len)):
            print(f"Trimming Pass {i+1}:\n\tWindow Length = {win_len[i]},\n\tHop Length = {hop_len[i]},\n\tReference db = {ref_db[i]},\n\tMargin Left = {margin_l[i]}")
        
        def func(array):
            process_wavs_multiprocess(array,
                filt_type=type,
                filt_cutoff_freq=cutoff_freq,
                filt_order=order,
                trim_margin_left=margin_l,
                trim_margin_right=margin_r,
                trim_top_db=ref_db,
                trim_window_length=win_len,
                trim_hop_length=hop_len,
                trim_ref=ref_f,
                trim_preemphasis_strength=empth,
                SAMPLE_RATE=SAMPLE_RATE,
                ignore_dirs=PATH_IGNORE+["VCTK-Corpus-0.92",],
                )
        multiprocess_directory_flacs(func, DIRECTORY_GLOBAL, threads=THREADS)
    
    if 0:
        print("Trimming VCTK")
        #        Pass |  1  |  2  |  3  |
        type        = ['hp' ,'hp' ,'lp' ]
        cutoff_freq = [150  ,40   ,18000]
        order       = [4    ,9    ,9    ]
        #    Pass |  1  |  2  |  3  |  4  |  5  |  6  |
        ref_db   =[60   ,44   ,40   ,40   ,42   ,44   ]
        win_len  =[16000,9600 ,4800 ,2400 ,1200 ,600  ]
        hop_len  =[1800 ,1200 ,600  ,300  ,150  ,150  ]
        ref_f    =[np.amax]*6
        empth    =[0.0]*6
        margin_l =[0,] + [0]*5
        margin_r =[0,] + [0]*5
        def func(array):
            process_wavs_multiprocess(array,
                filt_type=type,
                filt_cutoff_freq=cutoff_freq,
                filt_order=order,
                trim_margin_left=margin_l,
                trim_margin_right=margin_r,
                trim_top_db=ref_db,
                trim_window_length=win_len,
                trim_hop_length=hop_len,
                trim_ref=ref_f,
                trim_preemphasis_strength=empth,
                SAMPLE_RATE=SAMPLE_RATE,
                ignore_dirs=PATH_IGNORE,
                )
        multiprocess_directory_flacs(func, VCTK_directory, threads=THREADS)
    
    if 0: # normalize volumes
        print("Normalizing Volume of Each Folder to Match.")
        # Normalize Volume per Folder. ~~Should~~ Preserve most of the dynamic range and individual speaker identity while allowing multiple datasets to mix.
        normalize_wav_volumes_mixmode(DIRECTORY_GLOBAL)
    
# ---------Write metadata---------
    # Map wavs/text to training data/validation data files
    if 0:
        print("Building Nancy Metadata")
        nancy_build_metadata(Nancy_directory, Nancy_Prompts, SAMPLE_RATE=SAMPLE_RATE, BIT_DEPTH=AUDIO_DEPTH, min_audio_duration=MIN_DURATION, max_audio_duration=MAX_DURATION)
        
        print("Building MLP+VCTK Metadata")
        build_metadata(DIRECTORY_GLOBAL, ignore_dirs=PATH_IGNORE+[Nancy_directory,], SAMPLE_RATE=SAMPLE_RATE, BIT_DEPTH=AUDIO_DEPTH, min_audio_duration=MIN_DURATION, max_audio_duration=MAX_DURATION)
        
        print("Writing Metadata into filelists")
        write_datasets(permitted_noise_levels = [""], minimum_clips=MIN_CLIPS, start_token=START_TOKEN, end_token=STOP_TOKEN, percentage_training_data=PERCENT_TRAIN)
    
# ---------Generate Mel-Spectrograms---------
    os.system("cd '/media/cookie/Samsung PM961/TwiBot/tacotron2-PPP-1.3.0'; CUDA_VISIBLE_DEVICES=3 python3 generate_mels.py")

