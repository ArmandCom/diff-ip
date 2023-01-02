#%%

from PIL import Image
# from IPython.display import display
import torch as th

from composable_diffusion.download import download_model
from model_creation import (
	create_model_and_diffusion,
	model_and_diffusion_defaults,
	model_and_diffusion_defaults_upsampler
)

#%%

has_cuda = th.cuda.is_available()
device = th.device('cpu' if not has_cuda else 'cuda')

# Create base model.
timestep_respacing =  100 #@param{type: 'number'}
options = model_and_diffusion_defaults()

flags = {
	"image_size": 128,
	"num_channels": 192,
	"num_res_blocks": 2,
	"learn_sigma": True,
	"use_scale_shift_norm": False,
	"raw_unet": True,
	"noise_schedule": "squaredcos_cap_v2",
	"rescale_learned_sigmas": False,
	"rescale_timesteps": False,
	"num_classes": '2',
	"dataset": "clevr_pos_multiple",
	"use_fp16": has_cuda,
	"timestep_respacing": str(timestep_respacing)
}

for key, val in flags.items():
	options[key] = val

model, diffusion = create_model_and_diffusion(**options)
model.eval()
if has_cuda:
	model.convert_to_fp16()
model.to(device)
model.load_state_dict(th.load(download_model('clevr_pos'), device))
print('total base parameters', sum(x.numel() for x in model.parameters()))

#%%

# Sampling parameters
#@markdown `coordinates`: when composing  multiple positions, using `|` as the delimiter.
#@markdown Note: we use `x` in range `[0.1, 0.9]` and `y` in range `[0.25, 0.7]` since the training dataset labels are in given ranges.

coordinates = "0.1, 0.5 | 0.3, 0.5 | 0.5, 0.5 | 0.7, 0.5 | 0.9, 0.5" #@param{type: 'string'}
coordinates = [[float(x.split(',')[0].strip()), float(x.split(',')[1].strip())]
			   for x in coordinates.split('|')]
print(coordinates)
coordinates += [[-1, -1]] # add unconditional score label
batch_size = 1
guidance_scale = 10 #@param{type: 'number'}

#%%

# sampling function
def model_fn(x_t, ts, **kwargs):
	half = x_t[:1]
	combined = th.cat([half] * kwargs['y'].size(0), dim=0) # Note: Takes 1 element and repeats it for each condition. The non used conditions are set as unconditional model.
	model_out = model(combined, ts, **kwargs)
	eps, rest = model_out[:, :3], model_out[:, 3:]
	masks = kwargs.get('masks') # Note: Masks indicates which conditions are active. We could have it for all query list.
	cond_eps = eps[masks].mean(dim=0, keepdim=True) # Note: mean of contributions.
	uncond_eps = eps[~masks].mean(dim=0, keepdim=True) # Note: mean of unconditionals. Unnecessary
	half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
	eps = th.cat([half_eps] * x_t.size(0), dim=0)
	return th.cat([eps, rest], dim=1)

#%%

masks = [True] * (len(coordinates) - 1) + [False]
model_kwargs = dict(
	y=th.tensor(coordinates, dtype=th.float, device=device),
	masks=th.tensor(masks, dtype=th.bool, device=device)
)

#%%

##############################
# Sample from the base model #
##############################

# Create the text tokens to feed to the model.
def sample(coordinates):
	samples = diffusion.p_sample_loop(
		model_fn,
		(len(coordinates), 3, options["image_size"], options["image_size"]),
		device=device,
		clip_denoised=True,
		progress=True,
		model_kwargs=model_kwargs,
		cond_fn=None,
	)[:batch_size]

	# Show the output
	return samples

#%%

# Note: Obtain first data with their labels
#  Create a function to give you an answer of whether there's an object in a particular region.
#  Alternatively, select randomly X labels of an image as History.
#  Seems like in training there is a single object. Combine it using the masks. Masks can also be used to know the answer to our queries.
# sampling 128x128 images
samples = sample(coordinates)
show_images(samples)