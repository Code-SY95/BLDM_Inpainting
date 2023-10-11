# BLDM_Inpainting
### Pure Inpainting Code by using Blended Latent Diffusion Algorithm


```
python scripts/inpaint.py --indir <input image folder> --mask_scale <int,int,int,int> --outdir <output folder>
```

##### Essential args

- 'indir' is the name of the folder containing the input png images. Multiple images are also possible.


- 'mask_scale' is the masking scale (%) to be applied to each image. You can write the scale to be applied to each image in order as an integer from 0 to 100 with a ','. The integer should be the **same number** as the number of images in the 'indir'.
  -  For example, if you have 4 images in your 'indir' and want to apply a 'mask_scale' of 5%, 6%, 7%, and 8% to each of the 4 images in order, you would write 5,6,7,8 (without space).

  
- 'outdir' is the name of the folder where the images will be saved after seamless inpainting.


##### Optional args

  
- 'denoising_strength' is denoising strength of diffusion model. Default value is 0.3.


- 'steps' is the number of sampling steps. Default value is 50.
