# %%
import os
import glob
import PIL
import re
# %%
dir_in = './BaseGAN Pics/'
dir_out = './BaseGAN Pics/Gif/BaseGAN_gif.gif'
format = 'png'

files = glob.glob(os.path.join(dir_in, '*.' + format))
r = re.compile(r'\bfakes(\d+)..(\d+)\)\.' + format + r'\b')
sort_key = [(int(r.search(x).group(1)) * 100) + int(r.search(x).group(2)) for x in sorted(files)]
img, *imgs = [PIL.Image.open(x) for _, x in sorted(zip(sort_key, files))]
img.save(fp=dir_out, format='GIF', append_images=imgs, save_all=True, duration=100, loop=0)