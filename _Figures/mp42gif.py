#!/usr/bin/env python

import sys
import os
import shutil
import subprocess
import argparse

# Group of Different functions for different styles
if sys.platform.lower() == "win32":
    os.system('color')
RED = '\033[31m'
GREEN = '\033[32m'
RESET = '\033[0m'

def ffmpeg(mp4_video_filename):
    if not os.path.isfile(mp4_video_filename):
        raise Exception ('Missing MP4 video file: %s' % mp4_video_filename)
    filename, file_extension = os.path.splitext(mp4_video_filename)
    if file_extension != '.mp4':
        raise Exception ('Not MP4 video file: filename must have .mp4 extension. '\
                         'File extension is: %s' % file_extension)
    print(GREEN + 'Calling ffmpeg on MP4 video: {}'.format(mp4_video_filename) + RESET)
    try:
        subprocess.call('ffmpeg -i {} -vf scale=iw/2:ih/2:flags=lanczos,fps=10 frames/ffout%03d.png'.format(mp4_video_filename), shell = True)
        # subprocess.call('ffmpeg -i {} -vf scale=720:-1:flags=lanczos,fps=10 frames/ffout%03d.png'.format(mp4_video_filename), shell = True)
    except:
        raise Exception (RED + 'ffmpeg failed.' + RESET)
    return True

def clean():
    if os.path.exists('./frames'):
        print(GREEN + 'Cleaning generated frames folder. Use --keep_frames flag to override this.' + RESET)
        try:
            shutil.rmtree('./frames')
        except:
            raise Exception (RED + 'Removing frames folder failed.' + RESET)
    else:
        print(RED + 'Folder frames was not generated...' + RESET)
    return True

def generate_gif(out_name='output'):
    print(GREEN + 'Calling gm convert to generate {}.gif:'.format(out_name) + RESET)
    try:
        subprocess.call('gm convert -loop 0 ./frames/ffout*.png {}.gif'.format(out_name), shell=True)
    except:
        raise Exception (RED + 'gm convert failed: gif not generated.' + RESET)
    return True

def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    return arg

def parser():
    import argparse
    basic_desc = "Convert mp4 file to high-quality gif: use `mp4-to-gif VIDEO_FILENAME.mp4` \
            to convert the video to an `output.gif`. Use `--keep_frames` flag if you want to keep \
            intermediate frames: `mp4-to-gif VIDEO_FILENAME.mp4 --keep_frames"
    parser = argparse.ArgumentParser(add_help=True, description="{}".format(basic_desc))
    parser.add_argument("mp4_video_filename", help="Filename for MP4 video.", metavar="FILE",
                        type=lambda x: is_valid_file(parser, x))
    parser.add_argument("-k", "--keep_frames", action="store_true", help="Keep generated frames.",)
    return parser

if __name__ == "__main__":
    # Parse command line flags
    args = parser().parse_args()

    if not os.path.isdir('frames'):
        # If there is no frames folder, create one, and generate intermediate frames.
        print(GREEN + 'Creating frames folder.' + RESET)
        os.mkdir('frames')
        assert(ffmpeg(args.mp4_video_filename))
    else:
        # If we found a frames folder, use it to generate gif from inner frames.
        print(GREEN + 'Found existing frames folder.' + RESET)

    assert(generate_gif(args.mp4_video_filename.split('.mp4')[0]))
    if not args.keep_frames:
        assert(clean())
