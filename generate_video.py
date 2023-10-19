import os
import sys

# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
sys.path.extend([os.getcwd() + '/src/'])

import argparse
import torch
import clip
import random
from ViPE.utils import dotdict, get_lyrtic2prompts, get_track_intensity, get_visual_effects, get_visual_effects_disco
from ViPE.utils import add_audio_to_mp4, add_captions_to_video
import subprocess
import time, gc, os, sys
from types import SimpleNamespace
from helpers.save_images import get_output_folder
from helpers.settings import load_args
from helpers.render import render_animation, render_input_video, render_image_batch, render_interpolation
from helpers.model_load import load_model, get_model_output_paths
from helpers.aesthetics import load_aesthetics_model


def parse_args():
    parser = argparse.ArgumentParser(description="arguments for mp3 to video generation")

    parser.add_argument(
        "--vipe_checkpoint", type=str, default='fittar/ViPE-M-CTX7',
        help="which version of vipe to fetch from huggingface?"
    )

    parser.add_argument(
        "--mp3_file", type=str, help='name of the mp3 file', required=True
    )

    parser.add_argument(
        "--saving_dir", type=str, required=True, help='where to store the video and the required models'
    )

    parser.add_argument(
        "--music_gap_prompt", type=str, default='music notes',
        help="a prompt for nonvocal portions of the song/story"
    )
    parser.add_argument(
        "--music_gap_threshold", type=int, default=10,
        help='nonvocal interval in seconds for music_gap_prompt to be valid '
    )
    parser.add_argument(
        "--prefix", type=str, default=None,
        help="the overall theme of the song/story, be careful, it might has a strong effect on the video"
    )
    parser.add_argument(
        "--context_size", type=int, default=1, help='how many sentences to look back while interpreting the lyrics'
    )

    parser.add_argument(
        "--abstractness", type=float, default=.7, help='a real number between 0 and 1, how abstract the song/story is?'
    )
    parser.add_argument("--skip_vipe", action="store_true", help="skip using ViPE for prompt generation")
    parser.add_argument(
        "--image_quality_number", type=int, default=1,
        help='how many images to generate for each frame, the best image will be selected'
    )
    parser.add_argument(
        "--visual_effect_period", type=int, default=3,
        help='how many seconds each effect (a combination of camera movements) should last, not valid for disco mode)'
    )

    parser.add_argument(
        "--caption_mode", type=str, default=None,
        help='set to lyrics to add the lyrics, set to both for lyrics + vipe prompts'
    )
    parser.add_argument("--skip_visual_effect", action="store_true", help="pass the flag to skip having camera movements")

    parser.add_argument(
        "--animation_mode", type=str, default='3D',
        help='set to 2D for 2D animation'
    )
    parser.add_argument("--disco_mode", action="store_true", help="pass the flag to switch to disco mode")

    user_args = parser.parse_args()
    return user_args


def main():
    t0 = time.time()
    # print('job is running')
    user_args = parse_args()

    my_args = dotdict({})
    my_args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    mp3_dir = './mp3/'
    mp3_name = user_args.mp3_file
    my_args.saving_dir = user_args.saving_dir
    # '/graphics/scratch2/staff/hassan/test/'
    my_args.music_gap_prompt = user_args.music_gap_prompt
    my_args.prefix = user_args.prefix
    my_args.mp3_file = mp3_dir + '{}.mp3'.format(mp3_name)
    my_args.transcription_file = '{}{}_transcription'.format(mp3_dir, mp3_name)
    my_args.context_size = user_args.context_size
    my_args.song_abstractness = user_args.abstractness
    my_args.music_gap_threshold = user_args.music_gap_threshold  # seconds
    my_args.do_sample = True  # generate prompts using ViPE with sampling
    my_args.use_vipe = False if user_args.skip_vipe else True
    my_args.n_img_reward_samples = user_args.image_quality_number
    my_args.caption_mode = user_args.caption_mode  # set to None to skip adding lyrics, set to 'lyrics' to only add lyrics and 'both' to add both lyrics and prompts
    my_args.postfix_prompts = ", extreme detail, high quality, HD, 32K, dramatic lighting, ultra-realistic, high detailed photography, vivid, vibrant, intricate, trending on artstation"
    my_args.prompt_file = '{}/{}_ctx_{}_sample_{}_vipe_{}_abst_{}_lyric2prompt'.format(mp3_dir, mp3_name,
                                                                                       my_args.context_size,
                                                                                       my_args.do_sample,
                                                                                       my_args.use_vipe,
                                                                                       my_args.song_abstractness)
    my_args.disco_mode = True if user_args.disco_mode else False
    my_args.animation_mode = user_args.animation_mode
    my_args.use_init = False
    my_args.use_visual_effect = False if user_args.skip_visual_effect else True
    my_args.checkpoint = user_args.vipe_checkpoint

    fps_p = 15  # generate fps_p frames per seconds for each prompts
    visual_affect_chunk = user_args.visual_effect_period  # for how many seconds each visualization affect should last
    pass_render = False  # skip creating frames and make the video out the frames
    my_args.timestring = 'None'

    lyric2prompt = get_lyrtic2prompts(my_args)
    torch.cuda.empty_cache()

    animation_prompts = {}
    name = 'test_{}_rews_{}_{}fps_{}ctx_{}_vipe_{}_abst_{}'.format(my_args.animation_mode, my_args.n_img_reward_samples,
                                                                   fps_p, my_args.context_size, mp3_name,
                                                                   my_args.use_vipe,
                                                                   my_args.song_abstractness)
    for num, l2p in enumerate(lyric2prompt):
        # end = int(l2p['end'] * fps_p)
        start = int(l2p['start'] * fps_p)
        animation_prompts[start] = l2p['prompt'] + my_args.postfix_prompts

    if my_args.disco_mode:
        visual_effects = get_visual_effects_disco(my_args.mp3_file, fps_p, my_args.animation_mode)
    else:
        audio_intensity = get_track_intensity(my_args.mp3_file)
        visual_effects = get_visual_effects(audio_intensity, fps_p, visual_affect_chunk, my_args.animation_mode)

    def Root():
        saving_dir = my_args.saving_dir

        models_path = saving_dir + "models"  # @param {type:"string"}
        configs_path = saving_dir + "configs"  # @param {type:"string"}
        output_path = saving_dir + name  # @param {type:"string"}
        mount_google_drive = False  # @param {type:"boolean"}

        # @markdown **Model Setup**
        map_location = my_args.device   # @param ["cpu", "cuda"]
        # model_config = "v1-inference.yaml"  # @param ["custom","v2-inference.yaml","v2-inference-v.yaml","v1-inference.yaml"]
        model_config = "v1-inference.yaml"  # @param ["custom","v2-inference.yaml","v2-inference-v.yaml","v1-inference.yaml"]
        model_checkpoint = "Protogen_V2.2.ckpt"  # @param ["custom","v2-1_768-ema-pruned.ckpt","v2-1_512-ema-pruned.ckpt","768-v-ema.ckpt","512-base-ema.ckpt","Protogen_V2.2.ckpt","v1-5-pruned.ckpt","v1-5-pruned-emaonly.ckpt","sd-v1-4-full-ema.ckpt","sd-v1-4.ckpt","sd-v1-3-full-ema.ckpt","sd-v1-3.ckpt","sd-v1-2-full-ema.ckpt","sd-v1-2.ckpt","sd-v1-1-full-ema.ckpt","sd-v1-1.ckpt", "robo-diffusion-v1.ckpt","wd-v1-3-float16.ckpt"]
        custom_config_path = ""  # @param {type:"string"}
        custom_checkpoint_path = ""  # @param {type:"string"}
        return locals()

    root = Root()
    root = SimpleNamespace(**root)

    root.models_path, root.output_path = get_model_output_paths(root)
    root.model, root.device = load_model(root, load_on_run_all=True, check_sha256=True, map_location=root.map_location)

    def DeforumAnimArgs():
        # @markdown ####**Animation:**

        animation_mode = my_args.animation_mode  # @param ['None', '2D', '3D', 'Video Input', 'Interpolation'] {type:'string'}
        max_frames = int(lyric2prompt[-1]['end']) * fps_p  # @param {type:"number"}

        border = 'wrap'  # @param ['wrap', 'replicate'] {type:'string'}

        translation_z = "0:(0)"
        rotation_3d_x = "0:(0)"
        rotation_3d_y = "0:(0)"
        rotation_3d_z = "0:(0)"

        angle = "0:(0)"
        zoom = "0:(1)"
        translation_x = "0:(0)"
        translation_y = "0:(0)"

        if my_args.use_visual_effect:
            if my_args.animation_mode == '3D':
                translation_z = visual_effects['translation_z']  # @param {type:"string"}
                rotation_3d_x = visual_effects['rotation_3d_x']  # @param {type:"string"}
                rotation_3d_y = visual_effects['rotation_3d_y']  # @param {type:"string"}
                rotation_3d_z = visual_effects['rotation_3d_z']  # @param {type:"string"}

            else:
                angle = visual_effects['angles']  # @param {type:"string"}
                zoom = visual_effects['zooms']  # @param {type:"string"}
                translation_x = visual_effects['x_translations']  # @param {type:"string"}
                translation_y = visual_effects['y_translation']  # @param {type:"string"}

        flip_2d_perspective = False  # @param {type:"boolean"}
        perspective_flip_theta = "0:(0)"  # @param {type:"string"}
        perspective_flip_phi = "0:(t%15)"  # @param {type:"string"}
        perspective_flip_gamma = "0:(0)"  # @param {type:"string"}
        perspective_flip_fv = "0:(53)"  # @param {type:"string"}
        noise_schedule = "0: (0.02)"  # @param {type:"string"}
        if not my_args.use_init:
            strength_schedule = "0: (0.65)"  # @param {type:"string"}
        else:
            # use the first image for fps_p number of frames with low pompt strength
            strength_schedule = ""
            for stp in range(fps_p * 3):
                strength_schedule = strength_schedule + "{}: (0.97), ".format(stp)
            strength_schedule = strength_schedule + "{}: (0.65)".format(stp + 1)

        contrast_schedule = "0: (1.0)"  # @param {type:"string"}
        hybrid_video_comp_alpha_schedule = "0:(1)"  # @param {type:"string"}
        hybrid_video_comp_mask_blend_alpha_schedule = "0:(0.5)"  # @param {type:"string"}
        hybrid_video_comp_mask_contrast_schedule = "0:(1)"  # @param {type:"string"}
        hybrid_video_comp_mask_auto_contrast_cutoff_high_schedule = "0:(100)"  # @param {type:"string"}
        hybrid_video_comp_mask_auto_contrast_cutoff_low_schedule = "0:(0)"  # @param {type:"string"}

        # @markdown ####**Unsharp mask (anti-blur) Parameters:**
        kernel_schedule = "0: (5)"  # @param {type:"string"}
        sigma_schedule = "0: (1.0)"  # @param {type:"string"}
        amount_schedule = "0: (0.2)"  # @param {type:"string"}
        threshold_schedule = "0: (0.0)"  # @param {type:"string"}

        # @markdown ####**Coherence:**
        color_coherence = 'Match Frame 0 LAB'  # @param ['None', 'Match Frame 0 HSV', 'Match Frame 0 LAB', 'Match Frame 0 RGB', 'Video Input'] {type:'string'}
        color_coherence_video_every_N_frames = 1  # @param {type:"integer"}
        diffusion_cadence = '1'  # @param ['1','2','3','4','5','6','7','8'] {type:'string'}

        # @markdown ####**3D Depth Warping:**
        use_depth_warping = True  # @param {type:"boolean"}
        midas_weight = 0.3  # @param {type:"number"}
        near_plane = 200
        far_plane = 10000
        fov = 40  # @param {type:"number"}
        padding_mode = 'border'  # @param ['border', 'reflection', 'zeros'] {type:'string'}
        sampling_mode = 'bicubic'  # @param ['bicubic', 'bilinear', 'nearest'] {type:'string'}
        save_depth_maps = False  # @param {type:"boolean"}

        # @markdown ####**Video Input:**
        video_init_path = '/content/video_in.mp4'  # @param {type:"string"}
        extract_nth_frame = 1  # @param {type:"number"}
        overwrite_extracted_frames = True  # @param {type:"boolean"}
        use_mask_video = False  # @param {type:"boolean"}
        video_mask_path = '/content/video_in.mp4'  # @param {type:"string"}

        # @markdown ####**Hybrid Video for 2D/3D Animation Mode:**
        hybrid_video_generate_inputframes = False  # @param {type:"boolean"}
        hybrid_video_use_first_frame_as_init_image = True  # @param {type:"boolean"}
        hybrid_video_motion = "None"  # @param ['None','Optical Flow','Perspective','Affine']
        hybrid_video_flow_method = "Farneback"  # @param ['Farneback','DenseRLOF','SF']
        hybrid_video_composite = False  # @param {type:"boolean"}
        hybrid_video_comp_mask_type = "None"  # @param ['None', 'Depth', 'Video Depth', 'Blend', 'Difference']
        hybrid_video_comp_mask_inverse = False  # @param {type:"boolean"}
        hybrid_video_comp_mask_equalize = "None"  # @param  ['None','Before','After','Both']
        hybrid_video_comp_mask_auto_contrast = False  # @param {type:"boolean"}
        hybrid_video_comp_save_extra_frames = False  # @param {type:"boolean"}
        hybrid_video_use_video_as_mse_image = False  # @param {type:"boolean"}

        # @markdown ####**Interpolation:**
        interpolate_key_frames = False  # @param {type:"boolean"}
        interpolate_x_frames = 4  # @param {type:"number"}

        # @markdown ####**Resume Animation:**
        resume_from_timestring = False  # @param {type:"boolean"}
        # resume_timestring = "20230630115509"  # @param {type:"string"}

        return locals()

    override_settings_with_file = False  # @param {type:"boolean"}
    settings_file = "custom"  # @param ["custom", "512x512_aesthetic_0.json","512x512_aesthetic_1.json","512x512_colormatch_0.json","512x512_colormatch_1.json","512x512_colormatch_2.json","512x512_colormatch_3.json"]
    custom_settings_file = "/content/drive/MyDrive/Settings.txt"  # @param {type:"string"}

    def DeforumArgs():
        # @markdown **Image Settings**
        W = 512  # @param
        H = 512  # @param
        W, H = map(lambda x: x - x % 64, (W, H))  # resize to integer multiple of 64
        bit_depth_output = 8  # @param [8, 16, 32] {type:"raw"}
        n_img_reward_samples = my_args.n_img_reward_samples  # generate n images then select the best one based on imgreward method
        # @markdown **Sampling Settings**
        seed = -1  # @param
        sampler = 'euler_ancestral'  # @param ["klms","dpm2","dpm2_ancestral","heun","euler","euler_ancestral","plms", "ddim", "dpm_fast", "dpm_adaptive", "dpmpp_2s_a", "dpmpp_2m"]
        steps = 50  # @param
        scale = 7  # @param previosuly 7
        ddim_eta = 0.0  # @paramgra
        dynamic_threshold = None
        static_threshold = None

        # @markdown **Save & Display Settings**
        save_samples = True  # @param {type:"boolean"}
        save_settings = True  # @param {type:"boolean"}
        display_samples = True  # @param {type:"boolean"}
        save_sample_per_step = False  # @param {type:"boolean"}
        show_sample_per_step = False  # @param {type:"boolean"}

        # @markdown **Prompt Settings**
        prompt_weighting = True  # @param {type:"boolean"}
        normalize_prompt_weights = True  # @param {type:"boolean"}
        log_weighted_subprompts = False  # @param {type:"boolean"}

        # @markdown **Batch Settings**
        n_batch = 1  # @param
        batch_name = "ViPE"  # @param {type:"string"}
        filename_format = "{timestring}_{index}_{prompt}.png"  # @param ["{timestring}_{index}_{seed}.png","{timestring}_{index}_{prompt}.png"]
        seed_behavior = "iter"  # @param ["iter","fixed","random","ladder","alternate"]
        seed_iter_N = 1  # @param {type:'integer'}
        make_grid = False  # @param {type:"boolean"}
        grid_rows = 2  # @param
        outdir = get_output_folder(root.output_path, batch_name)

        # @markdown **Init Settings**
        use_init = my_args.use_init  # @param {type:"boolean"}
        strength = 1  # @param {type:"number"}
        strength_0_no_init = True  # Set the strength to 0 automatically when no init image is used
        init_image = "./ViPE/mp3/jaklin.jpg"  # @param {type:"string"}
        # Whiter areas of the mask are areas that change more
        use_mask = False  # @param {type:"boolean"}
        use_alpha_as_mask = False  # use the alpha channel of the init image as the mask
        mask_file = "https://www.filterforge.com/wiki/images/archive/b/b7/20080927223728%21Polygonal_gradient_thumb.jpg"  # @param {type:"string"}
        invert_mask = False  # @param {type:"boolean"}
        # Adjust mask image, 1.0 is no adjustment. Should be positive numbers.
        mask_brightness_adjust = 1.0  # @param {type:"number"}
        mask_contrast_adjust = 1.0  # @param {type:"number"}
        # Overlay the masked image at the end of the generation so it does not get degraded by encoding and decoding
        overlay_mask = True  # {type:"boolean"}
        # Blur edges of final overlay mask, if used. Minimum = 0 (no blur)
        mask_overlay_blur = 5  # {type:"number"}

        # @markdown **Exposure/Contrast Conditional Settings**
        mean_scale = 0  # @param {type:"number"}
        var_scale = 0  # @param {type:"number"}
        exposure_scale = 0  # @param {type:"number"}
        exposure_target = 0.5  # @param {type:"number"}

        # @markdown **Color Match Conditional Settings**
        colormatch_scale = 0  # @param {type:"number"}
        colormatch_image = "https://www.saasdesign.io/wp-content/uploads/2021/02/palette-3-min-980x588.png"  # @param {type:"string"}
        colormatch_n_colors = 4  # @param {type:"number"}
        ignore_sat_weight = 0  # @param {type:"number"}

        # @markdown **CLIP\Aesthetics Conditional Settings**
        clip_name = 'ViT-L/14'  # @param ['ViT-L/14', 'ViT-L/14@336px', 'ViT-B/16', 'ViT-B/32']
        clip_scale = 0  # @param {type:"number"}
        aesthetics_scale = 0  # @param {type:"number"}
        cutn = 1  # @param {type:"number"}
        cut_pow = 0.0001  # @param {type:"number"}

        # @markdown **Other Conditional Settings**
        init_mse_scale = 0  # @param {type:"number"}
        init_mse_image = "https://cdn.pixabay.com/photo/2022/07/30/13/10/green-longhorn-beetle-7353749_1280.jpg"  # @param {type:"string"}

        blue_scale = 0  # @param {type:"number"}

        # @markdown **Conditional Gradient Settings**
        gradient_wrt = 'x0_pred'  # @param ["x", "x0_pred"]
        gradient_add_to = 'both'  # @param ["cond", "uncond", "both"]
        decode_method = 'linear'  # @param ["autoencoder","linear"]
        grad_threshold_type = 'dynamic'  # @param ["dynamic", "static", "mean", "schedule"]
        clamp_grad_threshold = 0.2  # @param {type:"number"}
        clamp_start = 0.2  # @param
        clamp_stop = 0.01  # @param
        grad_inject_timing = list(range(1, 10))  # @param

        # @markdown **Speed vs VRAM Settings**
        cond_uncond_sync = True  # @param {type:"boolean"}

        n_samples = 1  # doesnt do anything
        precision = 'autocast'
        C = 4
        f = 8

        prompt = ""
        timestring = ""
        init_latent = None
        init_sample = None
        init_sample_raw = None
        mask_sample = None
        init_c = None
        seed_internal = 0

        return locals()

    args_dict = DeforumArgs()
    anim_args_dict = DeforumAnimArgs()

    if override_settings_with_file:
        load_args(args_dict, anim_args_dict, settings_file, custom_settings_file, verbose=False)

    args = SimpleNamespace(**args_dict)
    anim_args = SimpleNamespace(**anim_args_dict)

    args.timestring = time.strftime('%Y%m%d%H%M%S')
    if pass_render:
        args.timestring = my_args.timestring

    args.strength = max(0.0, min(1.0, args.strength))

    # Load clip model if using clip guidance
    if (args.clip_scale > 0) or (args.aesthetics_scale > 0):
        root.clip_model = clip.load(args.clip_name, jit=False)[0].eval().requires_grad_(False).to(root.device)
        if (args.aesthetics_scale > 0):
            root.aesthetics_model = load_aesthetics_model(args, root)

    if args.seed == -1:
        args.seed = random.randint(0, 2 ** 32 - 1)
    if not args.use_init:
        args.init_image = None
    if args.sampler == 'plms' and (args.use_init or anim_args.animation_mode != 'None'):
        print(f"Init images aren't supported with PLMS yet, switching to KLMS")
        args.sampler = 'klms'
    if args.sampler != 'ddim':
        args.ddim_eta = 0

    if anim_args.animation_mode == 'None':
        anim_args.max_frames = 1
    elif anim_args.animation_mode == 'Video Input':
        args.use_init = True

    # clean up unused memory
    gc.collect()
    torch.cuda.empty_cache()

    # dispatch to appropriate renderer
    if anim_args.animation_mode == '2D' or anim_args.animation_mode == '3D':
        if not pass_render:
            render_animation(args, anim_args, animation_prompts, root)
        # render_animation(args, anim_args, animation_prompts, root)
    elif anim_args.animation_mode == 'Video Input':
        render_input_video(args, anim_args, animation_prompts, root)
    elif anim_args.animation_mode == 'Interpolation':
        render_interpolation(args, anim_args, animation_prompts, root)
    else:
        render_image_batch(args, animation_prompts, root)

    """
    # Create Video From Frames
    """

    skip_video_for_run_all = False  # @param {type: 'boolean'}
    fps = fps_p  # @param {type:"number"}
    use_manual_settings = False  # @param {type:"boolean"}
    render_steps = False  # @param {type: 'boolean'}
    path_name_modifier = "x0_pred"  # @param ["x0_pred","x"]
    make_gif = False
    bitdepth_extension = "exr" if args.bit_depth_output == 32 else "png"

    if skip_video_for_run_all == True:
        print('Skipping video creation, uncheck skip_video_for_run_all if you want to run it')
    else:

        if use_manual_settings:
            max_frames = "200"  # @param {type:"string"}
        else:
            if render_steps:  # render steps from a single image
                fname = f"{path_name_modifier}_%05d.png"
                all_step_dirs = [os.path.join(args.outdir, d) for d in os.listdir(args.outdir) if
                                 os.path.isdir(os.path.join(args.outdir, d))]
                newest_dir = max(all_step_dirs, key=os.path.getmtime)
                image_path = os.path.join(newest_dir, fname)
                print(f"Reading images from {image_path}")
                mp4_path = os.path.join(newest_dir, f"{args.timestring}_{path_name_modifier}.mp4")
                max_frames = str(args.steps)
            else:  # render images for a video
                image_path = os.path.join(args.outdir, f"{args.timestring}_%05d.{bitdepth_extension}")
                mp4_path = os.path.join(root.output_path, f"{mp3_name}_mute_.mp4")
                max_frames = str(anim_args.max_frames)

        # make video
        cmd = [
            'ffmpeg',
            '-y',
            '-vcodec', bitdepth_extension,
            '-r', str(fps),
            '-start_number', str(0),
            '-i', image_path,
            '-frames:v', max_frames,
            '-c:v', 'libx264',
            '-vf',
            f'fps={fps}',
            '-pix_fmt', 'yuv420p',
            '-crf', '17',
            '-preset', 'veryfast',
            '-pattern_type', 'sequence',
            mp4_path
        ]

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            print(stderr)
            raise RuntimeError(stderr)

    if my_args.caption_mode is not None:
        add_captions_to_video(mp4_path, lyric2prompt, os.path.join(root.output_path, f"{mp3_name}_lyrics.mp4"),
                              my_args.caption_mode, my_args.add_fittar)

        add_audio_to_mp4(os.path.join(root.output_path, f"{mp3_name}_lyrics.mp4"), my_args.mp3_file,
                         os.path.join(root.output_path, f"{mp3_name}.mp4"))
        print('done adding the lyrics, prompts, and audio')
    else:
        add_audio_to_mp4(os.path.join(root.output_path, f"{mp3_name}_mute_.mp4"), my_args.mp3_file,
                         os.path.join(root.output_path, f"{mp3_name}.mp4"))
    t1 = time.time()
    print('video generation took, ', (t1 - t0) / 60, ' mins')


if __name__ == "__main__":
    main()
