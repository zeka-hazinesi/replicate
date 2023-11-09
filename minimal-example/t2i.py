from diffusers import (
    StableDiffusionXLAdapterPipeline,StableDiffusionXLPipeline, T2IAdapter,
EulerAncestralDiscreteScheduler, AutoencoderKL, DPMSolverMultistepScheduler,
DPMSolverSinglestepScheduler)
from diffusers.utils import load_image, make_image_grid
from controlnet_aux.midas import MidasDetector
import torch

# setup
#adapter = T2IAdapter.from_pretrained(
#  "TencentARC/t2i-adapter-depth-midas-sdxl-1.0", torch_dtype=torch.float16, varient="fp16"
#).to("cuda")
model_id = 'epoyraz/juggernautXL'
euler_a = EulerAncestralDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
#dpm_karras = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler", use_karras_sigmas=True)
#vae=AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = StableDiffusionXLPipeline.from_pretrained(
    model_id, scheduler=euler_a, torch_dtype=torch.float16, variant="fp16", 
).to("cuda")
pipe.enable_xformers_memory_efficient_attention()
#line_detector = MidasDetector.from_pretrained("lllyasviel/Annotators").to("cuda")

#predict
#image = load_image("input2.jpg")
#image = line_detector(
#    image, detect_resolution=512, image_resolution=1024
#)

prompt = "(living room:2), Award-Winning Nature-Inspired Interior Photo: Very minimalistic and simple style inspired by nature. There is lots of sage green, natural colours and concrete. Lots of indoor plants. award winning interior photo, photograph, ultra photorealistic, photorealism, film still, smooth shading, daylight, hyper realistic, behance, halation, bloom"
negative_prompt = ""
gen_images = pipe(
    prompt=prompt,
    height=1024,
    width=1312,
#    image=image,
#    negative_prompt=negative_prompt,
    num_inference_steps=32,
#    adapter_conditioning_scale=1.0,
    guidance_scale=9.0,
    generator=torch.Generator(device="cuda").manual_seed(943165716)
).images[0]
gen_images.save('output.png')