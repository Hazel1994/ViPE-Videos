import whisper
import ffmpeg
import json
import numpy as np
import librosa
import pandas as pd
import os
import sys
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from .generation import generate_from_sentences
from .semantic_similarity_eval import mpnet_embed_class
from moviepy.editor import VideoFileClip, AudioFileClip
from googletrans import Translator

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def whisper_transcribe(  audio_fpath="audio.mp3", device='cuda'):
    model = whisper.load_model('large').to(device)
    whispers= model.transcribe(audio_fpath)
    return whispers

def prepare_ViPE(args):
    model = GPT2LMHeadModel.from_pretrained(args.checkpoint)
    model.to(args.device)
    if 'ViPE-M' in args.checkpoint:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    else:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def get_duration(mp3_file):
    return float(ffmpeg.probe(mp3_file)['format']['duration'])


#get a list of sentences and append them together based on the context size
def prepare_lyrics(lyrics, context_size, prefix):

    for _ in range(context_size+1):
        lyrics = ['null'] + lyrics
    lyrics = [lyrics[i:i + context_size+1] for i in range(len(lyrics) - context_size )]
    for c, text in enumerate(lyrics):
        text='; '.join([t.replace('.','') for t in text if t != 'null'])
        lyrics[c] =  text

    lyrics.pop(0)

    if prefix is not None:
        lyrics=[prefix +'; ' +i for i in lyrics]

    return lyrics

def translate_to_eng(transcription):

    translator = Translator()
    trs_transcription=[]
    for line in transcription['segments']:
        text = line['text']
        text = translator.translate(text,dest='en').text
        line['text'] = text
        trs_transcription.append(line)
    return trs_transcription

 # import pandas as pd
 #    data=pd.read_csv('/graphics/scratch/shahmoha/temp/ac_EN_ratings/concreteness.csv',header=0,delimiter='\t' )
 #    word2score={w:s for w ,s in zip(data['WORD'],data['RATING'])}
def get_concreteness(prompts, word2score):
    scores=[]
    for prompt in prompts:
        conc_scores=[word2score[w]/10 for w in prompt.split() if w in word2score]
        if len(conc_scores) < 1:
            scores.append(0.10)
        else:
            scores.append(np.mean(conc_scores))

    return scores

def get_lyrtic2prompts(args):

    # if the prompt_file does not exist, generate it again
    if not os.path.isfile(args.prompt_file):
        #check if we have the transcription_file already
        if not os.path.isfile(args.transcription_file):
            transcription = whisper_transcribe(args.mp3_file,args.device)

            if transcription['language'] != 'en':
                print('detected language {}, translating to English'.format( transcription['language']))
                transcription= translate_to_eng(transcription)
            else:
                transcription=transcription['segments']

            plain_trans=[]
            for seg in transcription:
                plain_trans.append({'start':seg['start'], 'end':seg['end'], 'text':seg['text']})

            with open(args.transcription_file, 'w') as file:
                json.dump(plain_trans,file, indent = 6)

        keep_asking = True
        while keep_asking:
            # Ask the user a question
            user_input = input(
                "\033[91m Please check the timestamps of the generated transcription.\n"
                "Enter 'yes' to continue if youre happy with the transcription, or 'no' to stop and manually correct the file.): \033[0m")
            # Check the user's response
            if user_input.lower() == "yes":
                keep_asking = False
                print("Continuing to generate video...")
                # Add your code here for the part you want to continue with
            elif user_input.lower() == "no":
                print("Stopping the program.")
                sys.exit()  # This will terminate the program
            else:
                print('incorrect answer!')

        # laod the transcription_file
        with open(args.transcription_file, 'r') as file:
            transcription= json.load(file)

        model, tokenizer=prepare_ViPE(args)

        #preapare_lyrics
        processed_lyrics = prepare_lyrics([line['text'] for line in transcription], args.context_size, args.prefix)
        #Add the prompts to the transcription
        processed_transcription=[]

        if args.song_abstractness is not None:

            # for semantic similarity
            mpnet_object = mpnet_embed_class(device='cuda', nli=True)
            # for concrteness
            data = pd.read_csv('/graphics/scratch/shahmoha/temp/ac_EN_ratings/concreteness.csv', header=0,
                               delimiter='\t')
            word2score = {w: s for w, s in zip(data['WORD'], data['RATING'])}

        for i, lines in enumerate(transcription):

            if args.use_vipe:
                batch_size = random.choice([20, 40, 60])  # size of prompt galley per line of lyrics
                #batch_size = 100
                print('generating prompt using ViPE {} out of {}'.format(i+1, len(transcription)))
                lyric = processed_lyrics[i]
                # prompt = generate_from_sentences([lyric], model, tokenizer, hparams.device, do_sample)[
                #     0].replace(lyric, '')
               # prompts=generate_from_sentences([lyric] * batch_size, model, tokenizer,device=args.device, do_sample=args.do_sample,top_k=100, epsilon_cutoff=.00005, temperature=.6)

                if args.song_abstractness is not None:
                    prompts = generate_from_sentences([lyric] * batch_size, model, tokenizer, device=args.device,
                                                      do_sample=args.do_sample, top_k=None, epsilon_cutoff=None,
                                                      temperature=1)

                    # similarities=get_mpnet_embed_batch(prompts, [lyric] * batch_size, device='cuda', batch_size=batch_size).cpu().numpy()
                    # concreteness_score = get_concreteness(prompts)

                    similarities = mpnet_object.get_mpnet_embed_batch(prompts, [lyric] * batch_size,
                                                                      batch_size=batch_size).cpu().numpy()
                    concreteness_score = get_concreteness(prompts, word2score)

                    final_scores =[i* (1- args.song_abstractness) + (args.song_abstractness) * j   for i,j in zip(similarities,concreteness_score) ]
                    best_index=np.argmax(final_scores)
                    best_prompt=prompts[best_index]
                else:
                    best_prompt = generate_from_sentences([prompts] * 1, lyric, tokenizer,
                                                               device=args.device,
                                                               do_sample=True, top_k=None, epsilon_cutoff=None,
                                                               temperature=1)[0]
                # print(best_prompt)

                # print(lyric)
                # print('\n')
                # print(best_prompt)
                # print('\n\n')

                lines['prompt'] = best_prompt
            else:
                lines['prompt']=processed_lyrics[i]

            #lines['text'] = lyric

            processed_transcription.append(lines)

            # Add music gap if necessary
            if i+1 < len(transcription):

                gap_line = {}

                current_gap = transcription[i]['end'] - transcription[i + 1]['start']
                if (len(transcription) > i + 1 and current_gap >= args.music_gap_threshold):
                    gap_line['text'], gap_line['prompt'] = args.music_gap_prompt, generate_from_sentences(
                        [args.music_gap_prompt], model,
                        tokenizer, device=args.device,
                        do_sample=args.do_sample)[0]

                    gap_line['start'] = transcription[i]['end'] + current_gap / 5
                    gap_line['end'] = transcription[i]['end'] + (4 / 5 * current_gap)

                    processed_transcription.append(gap_line)

        transcription=processed_transcription
        #Add start of the song if the lyrics don't start at the beginning
        x = {}
        if transcription[0]['start'] != 0.0:
            x['start'] = 0.0
            x['end'] = transcription[0]['start']
            x['text'], x['prompt'] = args.music_gap_prompt, generate_from_sentences([args.music_gap_prompt], model, tokenizer,device=args.device, do_sample=args.do_sample)[0]
            transcription.insert(0,x)

        #get the duration of the song
        song_length = get_duration(args.mp3_file)
        #In case bugs exist towards the end of the transcriptions
        if transcription[-1]['end'] != song_length:
            #transcription[-1]['start'] = transcription[-2]['end']
            transcription[-1]['end'] = song_length


        with open(args.prompt_file, 'w') as file:
            json.dump(transcription,file, indent = 6)

    with open(args.prompt_file, 'r') as file:
        lyric2prompt = json.load(file)

    return lyric2prompt


def compute_rms(audio, sr, start_time, duration):
    # Extract the segment from the loaded audio
    segment = audio[int(start_time * sr):int((start_time + duration) * sr)]

    # Calculate the root mean square (RMS) of the audio segment
    rms = librosa.feature.rms(y=segment)

    # Calculate the average intensity across the track segment
    average_intensity = rms.mean()

    return average_intensity

def get_track_intensity(mp3_file):
    # Load the audio track once
    track_path = mp3_file
    audio, sr = librosa.load(track_path)

    # rms_avg = librosa.feature.rms(y=audio)
    # # Calculate the average intensity across the track segment
    # rms_avg = rms_avg.mean()

    # Calculate the total duration of the track in seconds
    track_duration = len(audio) / sr

    # Specify the periods of length 1 second
    periods = []
    start_time = 0
    duration = 1

    while start_time + duration <= track_duration:
        intensity = compute_rms(audio, sr, start_time, duration)
        # periods.append(min(1, intensity/rms_avg))
        periods.append(intensity)
        start_time += duration

    max_rms = max(periods)
    periods = [i / max_rms for i in periods]

    return periods

import random


def find_closest_match(current_time,frames_to_time ):
    best_distance=np.inf
    best_index=0
    for index, t in enumerate(frames_to_time):
        distance=abs(t - current_time)
        if distance < best_distance:
            best_distance = distance
            best_index =index
    return best_index

def get_visual_effects_disco(filename, fps_p,mode):

    y, sr = librosa.load(filename)
    # Calculate MFCCs (Mel-frequency cepstral coefficients)
    n_mfcc = 8  # Number of MFCC coefficients set to 20
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    frames_to_time = librosa.frames_to_time(range(mfccs.shape[1]), sr=sr, hop_length=512)
    #normalize the values
    for i in range(mfccs.shape[0]):
        x= mfccs[i,:]
        mfccs[i,:]= ( x - np.mean(mfccs[i,:])) / (np.std(mfccs[i,:]))
        mfccs[i, :] = (x - min(mfccs[i, :])) / (max(mfccs[i, :]) -  min(mfccs[i, :]) )

    # energy_d = .5
    # zoominout_d = .5
    # twerk_d = .4
    # spinin_d = .5
    # tilt_d = .5
    # pan_d = .45
    # pan_right_d = .40
    # zoom_out_d = .65

    thresholds=[np.mean(mfccs[i, :]) for i in range(8)]

    energy_d =np.mean(mfccs[0, :])  - 2* np.std(mfccs[0, :])
    twerk_d = thresholds[1]
    zoominout_d = thresholds[2]
    tilt_d = thresholds[3]
    pan_d_left = thresholds[4]
    pan_d_right = thresholds[5]
    spinin_d =thresholds[6]
    zoom_out_d = thresholds[7]

    # the number of frames used to determine the effect for a single frame
    #effective_frame_length=round((len(frames_to_time) / frames_to_time[-1]) / fps_p) -1
    total_video_frames=frames_to_time[-1] * fps_p

    translation_z = []
    rotation_3d_x = []
    rotation_3d_y = []
    rotation_3d_z = []
    count =[0 for _ in range(8)]

    zoom_control = 0
    twerk_control = 0
    pan_control=0
    for f in range(round(total_video_frames)):
        current_time=f/fps_p
        index=find_closest_match(current_time, frames_to_time)
        freq_data = mfccs[:, index]

        zoom_base=.05 if mode == '3D' else 1.01
        pan_base=0.02
        angle_base=0.0

        zoom_value=zoom_base
        tilt_value=0
        pan_value=pan_base
        angle_value=angle_base

        
        if freq_data[0] > energy_d:

            if freq_data[1] > twerk_d or freq_data[2]> zoominout_d:
                if random.randint(0, 1) == 1:
                    if twerk_control<1:
                        angle_value = freq_data[2]
                        twerk_control +=1
                    else:
                        angle_value = -freq_data[1]
                        twerk_control = 0
                        
                    count[1] += 1
                else:
                    if mode == '2D':
                        possible_values = np.linspace(0, 0.03, 10)
                        zoom_value = 1 + possible_values[int(freq_data[2] * 9)]
                        if zoom_control > 1:
                            zoom_value = 2  - zoom_value
                            zoom_control = 0
                        zoom_control += 1

                    else:
                        possible_values=np.linspace(0, 1.3, 100)
                        zoom_value = possible_values[int(freq_data[2]*99)]
                        if zoom_control >1:
                            zoom_value = - zoom_value
                            zoom_control=0
                        zoom_control +=1
                    count[2] += 1

            if freq_data[3] > tilt_d:
                tilt_value = freq_data[3]
                if random.randint(0, 1) == 1:
                    tilt_value = -freq_data[3]
                count[3] += 1

            if freq_data[4] > pan_d_left:
                pan_value = - freq_data[4]
                if random.randint(0, 10) > 5:
                    pan_value = -pan_value
                count[4] += 1

            if freq_data[5] > pan_d_right and freq_data[4] <= pan_d_left:
                pan_value = freq_data[5]
                if random.randint(0, 10) > 5:
                    pan_value = -pan_value
                count[5] += 1

            if freq_data[6] > spinin_d and angle_value==angle_base and pan_value==pan_base :
                angle_value = - freq_data[6]  # spin in positve
                count[6] += 1

            if freq_data[7] > zoom_out_d and zoom_value == zoom_base:
                if mode == '2D':
                    possible_values = np.linspace(0, 0.03, 10)
                    zoom_value = 1 - possible_values[int(freq_data[7] * 9)]

                else:
                    possible_values = np.linspace(0, .7, 100)
                    zoom_value =  - possible_values[int(freq_data[7] * 99)]
                count[7] += 1

            # if mode == '2D':
            #     if zoom_value < 0:
            #         zoom_value = 2 + zoom_value + .2
            #     else:
            #         zoom_value = zoom_value - .2


            translation_z.append("({})".format(zoom_value))
            rotation_3d_x.append("({})".format(tilt_value))
            rotation_3d_y.append("({})".format(pan_value))
            rotation_3d_z.append("({})".format(angle_value))
        else:
            if mode == '2D':
                zoom_value=1
            else:
                zoom_value=0.0

            angle_value = 0.0
            tilt_value = 0.0

            pan_control += 1

            if pan_control > -1 and pan_control < 15:
                pan_value = 0.1
            else:
                pan_value = - pan_value
            if  pan_control ==30:
                pan_control=0

            translation_z.append("({})".format(zoom_value))
            rotation_3d_x.append("({})".format(tilt_value))
            rotation_3d_y.append("({})".format(pan_value))
            rotation_3d_z.append("({})".format(angle_value))

    #print(count)
    #zooms = ["(1.02)" if i%2==0 else "(0.98)" for i in range(len(zooms))]
    translation_z=", ".join(["{}:{}".format(i,j) for i,j in enumerate(translation_z)])
    rotation_3d_x=", ".join(["{}:{}".format(i,j) for i,j in enumerate(rotation_3d_x)])
    rotation_3d_y=", ".join(["{}:{}".format(i,j) for i,j in enumerate(rotation_3d_y)])
    rotation_3d_z=", ".join(["{}:{}".format(i,j) for i,j in enumerate(rotation_3d_z)])

    if mode=='2D':
        return {'zooms': translation_z, 'x_translations': rotation_3d_x, 'y_translation': rotation_3d_y,
                'angles': rotation_3d_z}

    return {'translation_z':translation_z, 'rotation_3d_x':rotation_3d_x, 'rotation_3d_y':rotation_3d_y, 'rotation_3d_z':rotation_3d_z}




def get_visual_effects(audio_intensity, fps_p,visual_affect_chunk,mode):
    audio_intensity_match=[]
    for f in audio_intensity:
        for _ in range(fps_p):
            audio_intensity_match.append(f)

    # setting visual affect automatically
    angles=[]
    tr_xs=[]
    tr_ys=[]
    zooms=[]

    segment_size = fps_p * visual_affect_chunk
    counters=[0,0,0,0]
    for f, index in enumerate(range(0,len(audio_intensity),visual_affect_chunk)):

        #get a segment of the intensity list corresponding the visual_affect_chunk seconds
        start_inx=f *visual_affect_chunk
        end_inx=start_inx + visual_affect_chunk
        if end_inx >= len(audio_intensity):
            end_inx=-1

        choices=np.random.uniform(.2,1,4)
        choices.sort()
        avg_intensity_segment=np.mean(audio_intensity[start_inx:end_inx])

        # rotation + zoom
        if avg_intensity_segment > choices[3] and counters[0] <2  :
            zoom_value = 1.035 #1.03
            angle_value = .6 #7
            counters[0] +=1

            if counters[0] == 2:
                for ci in [1,2,3]:
                    counters[ci]=0

            x_tr=0
            y_tr=0
            if random.randint(0, 1)==1:
                zoom_value, angle_value =  .97, -angle_value

        # heavy zoom + slight rotation
        elif avg_intensity_segment > choices[2]  and counters[1] <2:
            zoom_value = 1.03
            angle_value =.1
            x_tr = 0
            y_tr =0
            if random.randint(0, 1) == 1:
                zoom_value = .97

            counters[1] +=1

            if counters[1]==2:
                for ci in [0, 2, 3]:
                    counters[ci] = 0

        # diogonal move  + slight zoom
        elif avg_intensity_segment > choices[1] and  counters[2] < 2:
            zoom_value = 1.002
            angle_value = 0

            x_tr = .3
            y_tr = .3
            if random.randint(0, 1) == 1:
                x_tr, y_tr = -x_tr, -y_tr

            counters[2] += 1

            if counters[2] == 2:
                for ci in [0, 1, 3]:
                    counters[ci] = 0

        # rotation +  slight zoom
        elif  avg_intensity_segment > choices[0]  and  counters[3] < 2:
            zoom_value = 1.003
            angle_value = .2
            x_tr = 0
            y_tr = 0
            if random.randint(0, 1) == 1:
                angle_value = -angle_value

            counters[3] += 1

            if counters[3] == 2:
                for ci in [0, 1,2]:
                    counters[ci] = 0

        # normal zoom
        else:
            zoom_value = 1.01
            angle_value = 0
            x_tr = 0
            y_tr = 0
            # if random.randint(0, 1) == 1:
            #     zoom_value = .98

        if mode == '3D':
            if zoom_value > 1:
                zoom_value = (zoom_value - 1) * 30
            else:
                zoom_value = -(1 - zoom_value) * 30


        for _ in range(segment_size):
            # last segment usually exceed the length of the audio
            if len(angles) == len(audio_intensity_match):
                break
            angles.append("({})".format(angle_value))
            tr_xs.append("({})".format(x_tr))
            tr_ys.append("({})".format(y_tr))
            zooms.append("({})".format(zoom_value))

    #zooms = ["(1.02)" if i%2==0 else "(0.98)" for i in range(len(zooms))]
    zooms=", ".join(["{}:{}".format(i,j) for i,j in enumerate(zooms)])
    tr_xs=", ".join(["{}:{}".format(i,j) for i,j in enumerate(tr_xs)])
    tr_ys=", ".join(["{}:{}".format(i,j) for i,j in enumerate(tr_ys)])
    angles=", ".join(["{}:{}".format(i,j) for i,j in enumerate(angles)])

    if mode=='3D':
        return {'translation_z': zooms, 'rotation_3d_x': tr_xs, 'rotation_3d_y': tr_ys, 'rotation_3d_z': angles}

    return {'zooms':zooms, 'x_translations':tr_xs, 'y_translation':tr_ys, 'angles':angles}


from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip

import cv2
import numpy as np

# Function to wrap text into lines with a given number of words per line
def wrap_text_to_lines(text, max_words_per_line,caption_mode,Vipe=True):
    words = text.split()
    if caption_mode != 'lyrics':
        if Vipe:
            current_line='ViPE: '
        else:
            current_line = 'Lyric: '
    else:
        current_line=""

    lines = []


    for word in words:
        if len(current_line) + len(word)  <= max_words_per_line:
            current_line += word + " "
        else:
            lines.append(current_line)
            current_line = word + " "

    lines.append(current_line)

    return lines

# def add_captions_to_video(video_path, captions, output_path, caption_mode, max_words_per_line=10):
#     cap = cv2.VideoCapture(video_path)
#
#     # Get video properties
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#
#     # Calculate the height of the rectangle for the captions
#     rectangle_height = 100
#
#     # Calculate the position of the bottom rectangle and text within it
#     rectangle_y = height - rectangle_height
#
#     # Define the codec for the output video
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#
#     # Create the VideoWriter object
#     out = cv2.VideoWriter(output_path, fourcc, fps, (width, height + rectangle_height))
#
#     # Describe the type of font to be used.
#     font = 0
#
#     for num, caption in enumerate(captions):
#         start_time = caption['start']
#         if num + 1 < len(captions):
#             end_time = captions[num + 1]['start'] - 0.01
#         else:
#             end_time = caption['end']
#
#         # Seek to the start time of the caption
#         cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
#
#         while cap.isOpened():
#             ret, frame = cap.read()
#
#             if not ret:
#                 break
#
#             # Check if it's time to stop displaying the caption
#             if cap.get(cv2.CAP_PROP_POS_MSEC) >= end_time * 1000:
#                 break
#
#             # Create a black rectangle for captions
#             caption_rectangle = np.zeros((rectangle_height, width, 3), np.uint8)
#
#             # Wrap caption['text'] into lines with max_words_per_line words per line
#             text_lines = wrap_text_to_lines(caption['text'], max_words_per_line, caption_mode, False)
#
#             x = 10  # Position the text within the rectangle
#             y_text = rectangle_y + 30  # Start the text at the top of the rectangle
#
#             for line in text_lines:
#                 cv2.putText(caption_rectangle,
#                             line,
#                             (x, y_text),
#                             font, 0.5,
#                             (0, 255, 255),
#                             2,
#                             cv2.LINE_4)
#                 y_text += 30
#
#             # Concatenate the frame and the caption rectangle
#             frame_with_caption = np.vstack((frame, caption_rectangle))
#
#             # Write the frame with text and prompt to the output video
#             out.write(frame_with_caption)
#
#     # Release the objects
#     cap.release()
#     out.release()


def add_captions_to_video(video_path, captions, output_path,caption_mode, add_fittar,max_chars_per_line=53):
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec for the output video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    # Create the VideoWriter object
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # describe the type of font to be used.
    #font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    font = 0
    # Position of the captions at the bottom of the video
    # Calculate the height of the rectangle for the captions
    rectangle_height = 95
    x = 30
    y = height - rectangle_height + 15

    if caption_mode =='lyrics':
        rectangle_height = 60
        y = height - rectangle_height + 15


    for num, caption in enumerate(captions):
        start_time = caption['start']
        if num +1  < len(captions):
            end_time = captions[num+1]['start'] - .01
        else:
            end_time =caption['end']

        # Seek to the start time of the caption
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break
            # Check if it's time to stop displaying the caption
            if cap.get(cv2.CAP_PROP_POS_MSEC) >= end_time  * 1000:
                break

            # Create a black rectangle for captions
            caption_rectangle = np.zeros((rectangle_height, width, 3), np.uint8)
            # caption_rectangle[:, :, 0] = 192  # Blue channel
            # caption_rectangle[:, :, 1] = 192  # Green channel
            # caption_rectangle[:, :, 2] = 192  # Red channel
            # Append the black rectangle to the bottom of the frame
            frame = np.vstack((frame[:height - rectangle_height], caption_rectangle))

            y_prompt = y
            y_text = y
            boldness=1
            font_size=.5
            if caption_mode !='lyrics':
                font_size=.4
                boldness=1
                # Wrap caption['prompt'] into lines with max_words_per_line words per line
                prompt_lines = wrap_text_to_lines(caption['prompt'], max_chars_per_line,caption_mode, True)

                for line in prompt_lines:
                    cv2.putText(frame,
                                line,
                                (x, y_prompt),
                                font, font_size,
                                (0, 255, 255),
                                boldness,
                                cv2.LINE_4)
                    y_prompt += 18

                y_text = y_prompt

            # Wrap caption['text'] into lines with max_words_per_line words per line
            text_lines = wrap_text_to_lines(caption['text'], max_chars_per_line,caption_mode,False)

            for line in text_lines:
                cv2.putText(frame,
                            line,
                            (x, y_text),
                            font, font_size,
                            (0, 150, 250),
                            boldness,
                            cv2.LINE_4)
                y_text += 18


            if add_fittar:
                cv2.putText(frame,
                            'Instagram: Fittar.art',
                            (width - 140, height - 67),
                            font, .4,
                            (0, 255, 255),
                            boldness,
                            cv2.LINE_4)

            # Write the frame with text and prompt to the output video
            out.write(frame)

    # Release the objects
    cap.release()
    out.release()





def add_audio_to_mp4(mp4_path, mp3_path, output_path):

    # Load the video clip
    video_clip = VideoFileClip(mp4_path)

    # Load the audio clip
    audio_clip = AudioFileClip(mp3_path)

    # Set the audio of the video clip to the loaded audio clip
    video_clip = video_clip.set_audio(audio_clip)

    # Get the audio codec and bitrate from the input audio clip
    # Get the audio codec and bitrate from the input audio clip
    audio_info = audio_clip.reader.infos
    audio_codec = audio_info.get("codec_name")
    audio_bitrate = audio_info.get("bit_rate")
    print(audio_bitrate)
    #  audio_bitrate=audio_bitrate,

    # # Calculate the video bitrate based on file size and duration
    # video_file_size = os.path.getsize(mp4_path)
    # video_duration = video_clip.duration
    # video_bitrate = int(video_file_size / video_duration)
    #print(video_bitrate)
    # Save the video with the added audio
    video_clip.write_videofile(output_path, codec="libx264", audio_codec=audio_codec,
                            audio_bitrate=audio_bitrate)

    # Close the video and audio clips
    video_clip.close()
    audio_clip.close()
