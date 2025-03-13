import argparse
import gc
import hashlib
import json
import os
import shlex
import subprocess
from contextlib import suppress
from urllib.parse import urlparse, parse_qs

import gradio as gr
import librosa
import numpy as np
import soundfile as sf
import sox
import yt_dlp
from pedalboard import Pedalboard, Reverb, Compressor, HighpassFilter
from pedalboard.io import AudioFile
from pydub import AudioSegment
from audio_separator.separator import Separator
from rvc import Config, load_hubert, get_vc, rvc_infer

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mdxnet_models_dir = os.path.join(BASE_DIR, 'mdxnet_models')
rvc_models_dir = os.path.join(BASE_DIR, 'rvc_models')
output_dir = os.path.join(BASE_DIR, 'song_output')
cookies_path = os.path.join(BASE_DIR, 'configs/ytdl.txt')


def get_youtube_video_id(url, ignore_playlist=True):
    """
    Extract the YouTube video ID from various URL formats.
    
    Examples:
      http://youtu.be/SA2iWivDJiE
      http://www.youtube.com/watch?v=_oPAwA_Udwc&feature=feedu
      http://www.youtube.com/embed/SA2iWivDJiE
      http://www.youtube.com/v/SA2iWivDJiE?version=3&amp;hl=en_US
    """
    parsed_url = urlparse(url)
    hostname = parsed_url.hostname or ''
    path = parsed_url.path

    if hostname.lower() == 'youtu.be':
        return path.lstrip('/')

    if hostname.lower() in {'www.youtube.com', 'youtube.com', 'music.youtube.com'}:
        if not ignore_playlist:
            with suppress(KeyError):
                return parse_qs(parsed_url.query)['list'][0]
        if parsed_url.path == '/watch':
            return parse_qs(parsed_url.query).get('v', [None])[0]
        if parsed_url.path.startswith('/watch/'):
            return parsed_url.path.split('/')[1]
        if parsed_url.path.startswith('/embed/'):
            return parsed_url.path.split('/')[2]
        if parsed_url.path.startswith('/v/'):
            return parsed_url.path.split('/')[2]

    return None


import yt_dlp

def yt_download(link):
    """
    Download the audio from a YouTube link as an mp3 file with cookies.
    """
    ydl_opts = {
        'format': 'bestaudio',
        'outtmpl': '%(title)s.%(ext)s',
        'nocheckcertificate': True,
,
        'extractaudio': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3'
        }],
        'cookiefile': f"{cookies_path}"  # Add cookies for authentication
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(link, download=True)
        if result:
            download_path = ydl.prepare_filename(result).replace('.webm', '.mp3').replace('.m4a', '.mp3')
            return download_path
        else:
            return None


def display_progress(message, percent, is_webui, progress=None):
    """
    Display progress either via the provided progress callback or by printing.
    """
    if is_webui and progress is not None:
        progress(percent, desc=message)
    else:
        print(message)


def raise_exception(error_msg, is_webui):
    """
    Raise an exception. If running in a web UI, use gr.Error.
    """
    if is_webui:
        raise gr.Error(error_msg)
    else:
        raise Exception(error_msg)


def get_rvc_model(voice_model, is_webui):
    """
    Search the specified RVC model directory for the model (.pth) and index (.index) files.
    """
    rvc_model_filename, rvc_index_filename = None, None
    model_dir = os.path.join(rvc_models_dir, voice_model)
    if not os.path.exists(model_dir):
        raise_exception(f'Model directory {model_dir} does not exist.', is_webui)
    for file in os.listdir(model_dir):
        ext = os.path.splitext(file)[1]
        if ext == '.pth':
            rvc_model_filename = file
        if ext == '.index':
            rvc_index_filename = file

    if rvc_model_filename is None:
        error_msg = f'No model file exists in {model_dir}.'
        raise_exception(error_msg, is_webui)

    model_path = os.path.join(model_dir, rvc_model_filename)
    index_path = os.path.join(model_dir, rvc_index_filename) if rvc_index_filename else ''
    return model_path, index_path


def separation_uvr(filename, output):
    """
    Run the separation steps using different pre-trained models.
    Returns a tuple of four file paths:
      - vocals_no_reverb: The vocals after initial de-echo/de-reverb (used as intermediate vocals)
      - instrumental_path: The separated instrumental audio
      - main_vocals_dereverb: The lead vocals after final de-reverb processing
      - backup_vocals: The backup vocals extracted in the final stage
    """
    separator = Separator(output_dir=output)
    base_name = os.path.splitext(os.path.basename(filename))[0]

    instrumental_path = os.path.join(output, f'{base_name}_Instrumental.wav')
    initial_vocals = os.path.join(output, f'{base_name}_Vocals.wav')
    vocals_no_reverb = os.path.join(output, f'{base_name}_Vocals (No Reverb).wav')
    vocals_reverb = os.path.join(output, f'{base_name}_Vocals (Reverb).wav')
    main_vocals_dereverb = os.path.join(output, f'{base_name}_Vocals_Main_DeReverb.wav')
    backup_vocals = os.path.join(output, f'{base_name}_Vocals_Backup.wav')

    separator.load_model(model_filename='model_bs_roformer_ep_317_sdr_12.9755.ckpt')
    voc_inst = separator.separate(filename)
    os.rename(os.path.join(output, voc_inst[0]), instrumental_path)
    os.rename(os.path.join(output, voc_inst[1]), initial_vocals)

    separator.load_model(model_filename='UVR-DeEcho-DeReverb.pth')
    voc_no_reverb = separator.separate(initial_vocals)
    os.rename(os.path.join(output, voc_no_reverb[0]), vocals_no_reverb)
    os.rename(os.path.join(output, voc_no_reverb[1]), vocals_reverb)

    separator.load_model(model_filename='mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt')
    voc_split = separator.separate(vocals_no_reverb)
    os.rename(os.path.join(output, voc_split[0]), backup_vocals)
    os.rename(os.path.join(output, voc_split[1]), main_vocals_dereverb)

    if os.path.exists(vocals_reverb):
        os.remove(vocals_reverb)

    return vocals_no_reverb, instrumental_path, main_vocals_dereverb, backup_vocals


def get_audio_paths(song_dir):
    """
    Search the given directory for expected audio files.
    Returns:
      orig_song_path, instrumentals_path, main_vocals_dereverb_path, backup_vocals_path
    """
    orig_song_path = None
    instrumentals_path = None
    main_vocals_dereverb_path = None
    backup_vocals_path = None

    for file in os.listdir(song_dir):
        if file.endswith('_Instrumental.wav'):
            instrumentals_path = os.path.join(song_dir, file)
            orig_song_path = instrumentals_path.replace('_Instrumental', '')
        elif file.endswith('_Vocals_Main_DeReverb.wav'):
            main_vocals_dereverb_path = os.path.join(song_dir, file)
        elif file.endswith('_Vocals_Backup.wav'):
            backup_vocals_path = os.path.join(song_dir, file)

    return orig_song_path, instrumentals_path, main_vocals_dereverb_path, backup_vocals_path


def convert_to_stereo(audio_path):
    """
    Convert the given audio file to stereo (2 channels) if it is mono.
    """
    wave, sr = librosa.load(audio_path, mono=False, sr=44100)
    if wave.ndim == 1:
        stereo_path = f'{os.path.splitext(audio_path)[0]}_stereo.wav'
        command = shlex.split(f'ffmpeg -y -loglevel error -i "{audio_path}" -ac 2 -f wav "{stereo_path}"')
        subprocess.run(command, check=True)
        return stereo_path
    return audio_path


def pitch_shift(audio_path, pitch_change):
    """
    Shift the pitch of the audio by the specified amount.
    """
    output_path = f'{os.path.splitext(audio_path)[0]}_p{pitch_change}.wav'
    if not os.path.exists(output_path):
        y, sr = sf.read(audio_path)
        tfm = sox.Transformer()
        tfm.pitch(pitch_change)
        y_shifted = tfm.build_array(input_array=y, sample_rate_in=sr)
        sf.write(output_path, y_shifted, sr)
    return output_path


def get_hash(filepath):
    """
    Calculate a short BLAKE2b hash for the given file.
    """
    with open(filepath, 'rb') as f:
        file_hash = hashlib.blake2b()
        while chunk := f.read(8192):
            file_hash.update(chunk)
    return file_hash.hexdigest()[:11]


def preprocess_song(song_input, song_id, is_webui, input_type, progress):
    """
    Preprocess the input song:
      - Download if YouTube URL.
      - Convert to stereo.
      - Separate vocals and instrumentals.
    Returns a tuple with six values matching the expected unpacking in the pipeline.
    """
    if input_type == 'yt':
        display_progress('[~] Downloading song...', 0, is_webui, progress)
        song_link = song_input.split('&')[0]
        orig_song_path = yt_download(song_link)
    elif input_type == 'local':
        orig_song_path = song_input
    else:
        orig_song_path = None

    song_output_dir = os.path.join(output_dir, song_id)
    if not os.path.exists(song_output_dir):
        os.makedirs(song_output_dir)

    orig_song_path = convert_to_stereo(orig_song_path)

    display_progress('[~] Separating Vocals from Instrumental...', 0.1, is_webui, progress)
    vocals_no_reverb, instrumental_path, main_vocals_dereverb, backup_vocals = separation_uvr(orig_song_path, song_output_dir)
    return orig_song_path, vocals_no_reverb, instrumental_path, main_vocals_dereverb, backup_vocals, main_vocals_dereverb


def voice_change(voice_model, vocals_path, output_path, pitch_change, f0_method,
                 index_rate, filter_radius, rms_mix_rate, protect, crepe_hop_length, is_webui):
    """
    Convert the input vocals using the specified RVC model.
    """
    rvc_model_path, rvc_index_path = get_rvc_model(voice_model, is_webui)
    device = 'cuda:0'
    config = Config(device, True)
    hubert_model = load_hubert(device, config.is_half, os.path.join(rvc_models_dir, 'hubert_base.pt'))
    
    cpt, version, net_g, tgt_sr, vc = get_vc(device, config.is_half, config, rvc_model_path)

    rvc_infer(
        rvc_index_path, index_rate, vocals_path, output_path, pitch_change, f0_method,
        cpt, version, net_g, filter_radius, tgt_sr, rms_mix_rate, protect,
        crepe_hop_length, vc, hubert_model
    )
    del hubert_model, cpt
    gc.collect()


def add_audio_effects(audio_path, reverb_rm_size, reverb_wet, reverb_dry, reverb_damping):
    """
    Apply a chain of audio effects (highpass, compression, reverb) to the input audio.
    """
    output_path = f'{os.path.splitext(audio_path)[0]}_mixed.wav'
    board = Pedalboard([
        HighpassFilter(),
        Compressor(ratio=4, threshold_db=-15),
        Reverb(room_size=reverb_rm_size, dry_level=reverb_dry, wet_level=reverb_wet, damping=reverb_damping)
    ])

    with AudioFile(audio_path) as f:
        with AudioFile(output_path, 'w', f.samplerate, f.num_channels) as o:
            while f.tell() < f.frames:
                chunk = f.read(int(f.samplerate))
                effected = board(chunk, f.samplerate, reset=False)
                o.write(effected)
    return output_path


def combine_audio(audio_paths, output_path, main_gain, backup_gain, inst_gain, output_format):
    """
    Combine main vocals, backup vocals, and instrumental audio into a final mix.
    """
    main_vocal_audio = AudioSegment.from_wav(audio_paths[0]) - 4 + main_gain
    backup_vocal_audio = AudioSegment.from_wav(audio_paths[1]) - 6 + backup_gain
    instrumental_audio = AudioSegment.from_wav(audio_paths[2]) - 7 + inst_gain
    final_audio = main_vocal_audio.overlay(backup_vocal_audio).overlay(instrumental_audio)
    final_audio.export(output_path, format=output_format)


def song_cover_pipeline(song_input, voice_model, pitch_change, keep_files,
                        is_webui=0, main_gain=0, backup_gain=0, inst_gain=0, index_rate=0.5, filter_radius=3,
                        rms_mix_rate=0.25, f0_method='rmvpe', crepe_hop_length=128, protect=0.33, pitch_change_all=0,
                        reverb_rm_size=0.15, reverb_wet=0.2, reverb_dry=0.8, reverb_damping=0.7, output_format='mp3',
                        progress=gr.Progress()):
    """
    Main pipeline that orchestrates the AI cover song generation.
    """
    try:
        if not song_input or not voice_model:
            raise_exception('Ensure that the song input field and voice model field is filled.', is_webui)

        display_progress('[~] Starting AI Cover Generation Pipeline...', 0, is_webui, progress)

        if urlparse(song_input).scheme == 'https':
            input_type = 'yt'
            song_id = get_youtube_video_id(song_input)
            if song_id is None:
                raise_exception('Invalid YouTube url.', is_webui)
        else:
            input_type = 'local'
            song_input = song_input.strip('\"')
            if os.path.exists(song_input):
                song_id = get_hash(song_input)
            else:
                raise_exception(f'{song_input} does not exist.', is_webui)

        song_dir = os.path.join(output_dir, song_id)

        if not os.path.exists(song_dir):
            os.makedirs(song_dir)
            (orig_song_path, vocals_path, instrumentals_path,
             main_vocals_path, backup_vocals_path, main_vocals_dereverb_path) = preprocess_song(
                song_input, song_id, is_webui, input_type, progress
            )
        else:
            vocals_path, main_vocals_path = None, None
            paths = get_audio_paths(song_dir)
            if any(path is None for path in paths) or keep_files:
                (orig_song_path, vocals_path, instrumentals_path,
                 main_vocals_path, backup_vocals_path, main_vocals_dereverb_path) = preprocess_song(
                    song_input, song_id, is_webui, input_type, progress
                )
            else:
                orig_song_path, instrumentals_path, main_vocals_dereverb_path, backup_vocals_path = paths
                main_vocals_path = main_vocals_dereverb_path

        pitch_change += pitch_change_all

        base_song_name = os.path.splitext(os.path.basename(orig_song_path))[0]
        algo_suffix = f"_{crepe_hop_length}" if f0_method == "mangio-crepe" else ""
        ai_vocals_path = os.path.join(
            song_dir,
            f'{base_song_name}_lead_{voice_model}_p{pitch_change}_i{index_rate}_fr{filter_radius}_'
            f'rms{rms_mix_rate}_pro{protect}_{f0_method}{algo_suffix}.wav'
        )
        ai_backing_path = os.path.join(
            song_dir,
            f'{base_song_name}_backing_{voice_model}_p{pitch_change}_i{index_rate}_fr{filter_radius}_'
            f'rms{rms_mix_rate}_pro{protect}_{f0_method}{algo_suffix}.wav'
        )
        ai_cover_path = os.path.join(song_dir, f'{base_song_name} ({voice_model} Ver).{output_format}')
        ai_cover_backing_path = os.path.join(song_dir, f'{base_song_name} ({voice_model} Ver With Backing).{output_format}')

        if not os.path.exists(ai_vocals_path):
            display_progress('[~] Converting lead voice using RVC...', 0.5, is_webui, progress)
            voice_change(voice_model, main_vocals_dereverb_path, ai_vocals_path, pitch_change,
                         f0_method, index_rate, filter_radius, rms_mix_rate, protect, crepe_hop_length, is_webui)

            display_progress('[~] Converting backing voice using RVC...', 0.65, is_webui, progress)
            voice_change(voice_model, backup_vocals_path, ai_backing_path, pitch_change,
                         f0_method, index_rate, filter_radius, rms_mix_rate, protect, crepe_hop_length, is_webui)

        display_progress('[~] Applying audio effects to Vocals...', 0.8, is_webui, progress)
        ai_vocals_mixed_path = add_audio_effects(ai_vocals_path, reverb_rm_size, reverb_wet, reverb_dry, reverb_damping)
        ai_backing_mixed_path = add_audio_effects(ai_backing_path, reverb_rm_size, reverb_wet, reverb_dry, reverb_damping)

        if pitch_change_all != 0:
            display_progress('[~] Applying overall pitch change', 0.85, is_webui, progress)
            instrumentals_path = pitch_shift(instrumentals_path, pitch_change_all)
            backup_vocals_path = pitch_shift(backup_vocals_path, pitch_change_all)

        display_progress('[~] Combining AI Vocals and Instrumentals...', 0.9, is_webui, progress)
        combine_audio([ai_vocals_mixed_path, backup_vocals_path, instrumentals_path],
                      ai_cover_path, main_gain, backup_gain, inst_gain, output_format)
        combine_audio([ai_vocals_mixed_path, ai_backing_mixed_path, instrumentals_path],
                      ai_cover_backing_path, main_gain, backup_gain, inst_gain, output_format)

        if not keep_files:
            display_progress('[~] Removing intermediate audio files...', 0.95, is_webui, progress)
            intermediate_files = [vocals_path, main_vocals_path, ai_vocals_mixed_path, ai_backing_mixed_path]
            if pitch_change_all != 0:
                intermediate_files += [instrumentals_path, backup_vocals_path]
            for file in intermediate_files:
                if file and os.path.exists(file):
                    os.remove(file)

        return ai_cover_path, ai_cover_backing_path

    except Exception as e:
        raise_exception(str(e), is_webui)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='AICoverGen: Mod.',
        add_help=True
    )
    parser.add_argument('-i', '--song-input', type=str, required=True,
                        help='Link to a YouTube video or the filepath to a local mp3/wav file to create an AI cover of')
    parser.add_argument('-dir', '--rvc-dirname', type=str, required=True,
                        help='Name of the folder in the rvc_models directory containing the RVC model file and optional index file to use')
    parser.add_argument('-p', '--pitch-change', type=int, required=True,
                        help='Change the pitch of AI Vocals only. Generally, use 1 for male to female and -1 for vice-versa. (Octaves)')
    parser.add_argument('-k', '--keep-files', action=argparse.BooleanOptionalAction,
                        help='Whether to keep all intermediate audio files generated in the song_output/id directory, e.g. Isolated Vocals/Instrumentals')
    parser.add_argument('-ir', '--index-rate', type=float, default=0.5,
                        help='A decimal number e.g. 0.5, used to reduce/resolve the timbre leakage problem. If set to 1, more biased towards the timbre quality of the training dataset')
    parser.add_argument('-fr', '--filter-radius', type=int, default=3,
                        help='A number between 0 and 7. If >=3: apply median filtering to the harvested pitch results. The value represents the filter radius and can reduce breathiness.')
    parser.add_argument('-rms', '--rms-mix-rate', type=float, default=0.25,
                        help="A decimal number e.g. 0.25. Control how much to use the original vocal's loudness (0) or a fixed loudness (1).")
    parser.add_argument('-palgo', '--pitch-detection-algo', type=str, default='rmvpe',
                        help='Best option is rmvpe (clarity in vocals), then mangio-crepe (smoother vocals).')
    parser.add_argument('-hop', '--crepe-hop-length', type=int, default=128,
                        help='If pitch detection algo is mangio-crepe, controls how often it checks for pitch changes in milliseconds. Recommended: 128.')
    parser.add_argument('-pro', '--protect', type=float, default=0.33,
                        help='A decimal number e.g. 0.33. Protect voiceless consonants and breath sounds to prevent artifacts such as tearing in electronic music.')
    parser.add_argument('-mv', '--main-vol', type=int, default=0,
                        help='Volume change for AI main vocals in decibels. Use -3 to decrease by 3 dB and 3 to increase by 3 dB')
    parser.add_argument('-bv', '--backup-vol', type=int, default=0,
                        help='Volume change for backup vocals in decibels')
    parser.add_argument('-iv', '--inst-vol', type=int, default=0,
                        help='Volume change for instrumentals in decibels')
    parser.add_argument('-pall', '--pitch-change-all', type=int, default=0,
                        help='Change the pitch/key of vocals and instrumentals. Changing this slightly reduces sound quality')
    parser.add_argument('-rsize', '--reverb-size', type=float, default=0.15,
                        help='Reverb room size between 0 and 1')
    parser.add_argument('-rwet', '--reverb-wetness', type=float, default=0.2,
                        help='Reverb wet level between 0 and 1')
    parser.add_argument('-rdry', '--reverb-dryness', type=float, default=0.8,
                        help='Reverb dry level between 0 and 1')
    parser.add_argument('-rdamp', '--reverb-damping', type=float, default=0.7,
                        help='Reverb damping between 0 and 1')
    parser.add_argument('-oformat', '--output-format', type=str, default='mp3',
                        help='Output format of audio file. mp3 for smaller file size, wav for best quality')
    args = parser.parse_args()

    rvc_dir = os.path.join(rvc_models_dir, args.rvc_dirname)
    if not os.path.exists(rvc_dir):
        raise Exception(f'The folder {rvc_dir} does not exist.')

    cover_path, cover_with_backing = song_cover_pipeline(
        args.song_input, args.rvc_dirname, args.pitch_change, args.keep_files,
        main_gain=args.main_vol, backup_gain=args.backup_vol, inst_gain=args.inst_vol,
        index_rate=args.index_rate, filter_radius=args.filter_radius,
        rms_mix_rate=args.rms_mix_rate, f0_method=args.pitch_detection_algo,
        crepe_hop_length=args.crepe_hop_length, protect=args.protect,
        pitch_change_all=args.pitch_change_all,
        reverb_rm_size=args.reverb_size, reverb_wet=args.reverb_wetness,
        reverb_dry=args.reverb_dryness, reverb_damping=args.reverb_damping,
        output_format=args.output_format
    )
    print(f'[+] Cover generated at {cover_path}')
