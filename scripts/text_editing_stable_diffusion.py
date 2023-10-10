import argparse
import numpy as np
from PIL import Image

from diffusers import DDIMScheduler, StableDiffusionPipeline
import torch


class BlendedLatnetDiffusion:
    def __init__(self):
        self.parse_args()
        self.load_models()

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--prompt", type=str, required=True, help="The target text prompt"
        )
        parser.add_argument(
            "--init_image", type=str, required=True, help="The path to the input image"
        )
        parser.add_argument(
            "--mask", type=str, required=True, help="The path to the input mask"
        )
        parser.add_argument(
            "--model_path",
            type=str,
            default="stabilityai/stable-diffusion-2-1-base",
            help="The path to the HuggingFace model",
        )
        parser.add_argument(
            "--batch_size", type=int, default=1, help="The number of images to generate"
        )
        parser.add_argument(
            "--blending_start_percentage",
            type=float,
            default=0.25,
            help="The diffusion steps percentage to jump",
        )
        parser.add_argument("--device", type=str, default="cuda")
        parser.add_argument(
            "--output_path",
            type=str,
            default="outputs/res.jpg",
            help="The destination output path",
        )

        self.args = parser.parse_args()

    def load_models(self):
        pipe = StableDiffusionPipeline.from_pretrained(
            self.args.model_path, torch_dtype=torch.float16
        )
        
        self.vae = pipe.vae.to(self.args.device)
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder.to(self.args.device)
        self.unet = pipe.unet.to(self.args.device)
        self.scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )

    @torch.no_grad()
    def edit_image(
        self,
        image_path,
        mask_path,
        prompts,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        generator=torch.manual_seed(42),
        blending_percentage=0.25, # SY: denoising strength
    ):
        batch_size = len(prompts)

        image = Image.open(image_path)
        image = image.resize((height, width), Image.BILINEAR)
        image = np.array(image)[:, :, :3]
        source_latents = self._image2latent(image)
        latent_mask, org_mask = self._read_mask(mask_path)

        text_input = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to("cuda"))[0]

        max_length = text_input.input_ids.shape[-1]
        uncond_input = self.tokenizer(
            [""] * batch_size,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to("cuda"))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # Original bldm
        latents = torch.randn(
            (batch_size, self.unet.in_channels, height // 8, width // 8),
            generator=generator,
        ) # SY: self.unet.in_channels 9 = mask(=1) + masked_img(=4) + latent_input_img(=4)
        
        ## Sy eddited for pure inpainting
        # latents = torch.randn(
        #     (batch_size, 4, height // 8, width // 8),
        #     generator=generator,
        # )
        
        latents = latents.to("cuda").half() # Sy: [b,u,64,64]. (u = 9 : cond = "hybrid") / (u = 5? : cond = "concat")
        '''
        cond = "hybrid"
            - input image(4) : x = [b,4,64,64] (because the latent chanel size is 4?)
            - c_concat(5) = mask(1) + masked image(4) : [b,5,64,64]
            - cc = prompt
            - out = self.scripted_diffusion_model(xc, t, context=cc)
        cond = "concat" 
            - input image(4) : x = [b,4,64,64]
            - c_concat(5) = mask(1) + masked image(4) : [b,5,64,64] ??
            - out = self.diffusion_model(xc, t)
        '''

        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.scheduler.timesteps[
            int(len(self.scheduler.timesteps) * blending_percentage) :
        ]:
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2) # SY: torch.Size([2b, u, 64, 64])

            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, timestep=t
            ) # SY: self.scheduler.scale_model_input(latent_model_input, timestep=t) = latent_model_input = torch.Size([2b, 9, 64, 64])

            '''
            <Edited part>
                - latent mask : [b,1,64,64]
                    - source_latents : [b,4,64,64]
                - latent masked image = mask+input image (for background) : [b,4,64,64]
                - latent model input (to UNet) : [b,u,64,64]
            '''
            # latent_mask_batch = torch.cat([latent_mask] * 2)

            # masked_image_latents = source_latents * (latent_mask  < 0.5) #MJ: get the background image
            # c_concat = torch.cat([latent_mask, masked_image_latents], dim=1)
            # c_concat_batch = torch.cat([c_concat] * 2)
            #MJ: Add this line to the original blended-latent-diffusion, which uses
            # StableDiffusionPipeline (txt2imge), which has num_channels_unet=self.unet.config.in_channels = 9
            # latent_model_input = torch.cat([latent_model_input, c_concat_batch], dim=1)
            #MJ: latent_mask ([b,1,64,64]) and masked_image_latents (b,4,64,64]) are broadcasted to latent_model_input ([2b,9,64,64])??

            # predict the noise residual
            with torch.no_grad():
                noise_pred = self.unet(  # Sy: self.unet 's lasy out layer: Conv2d(320, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                    latent_model_input, t, encoder_hidden_states=text_embeddings
                ).sample # SY: torch.Size([2b, 4, 64, 64])
                # self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2) # SY: One each torch.Size([b, 4, 64, 64])
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            ) # SY: 
            # noise_pred_text - noise_pred_uncond = zero tensor. (Because of empty prompt)
            # noise_pred = torch.Size([b, 4, 64, 64])

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            # SY
            # noise_pred.shape = torch.Size([b, 4, 64, 64]) | latents.shape = torch.Size([b, 9, 64, 64])
            # t = tensor(680)

            # Blending
            # Sy : z_bg ~ noise(z_init, t), z_init= source_latents: z_bg = noise_source_latents
            noise_source_latents = self.scheduler.add_noise(
                source_latents, torch.randn_like(latents), t
            )
            latents = latents * latent_mask + noise_source_latents * (1 - latent_mask)

        latents = 1 / 0.18215 * latents

        with torch.no_grad():
            image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")

        return images

    @torch.no_grad()
    def _image2latent(self, image):
        image = torch.from_numpy(image).float() / 127.5 - 1
        image = image.permute(2, 0, 1).unsqueeze(0).to("cuda")
        image = image.half()
        latents = self.vae.encode(image)["latent_dist"].mean
        latents = latents * 0.18215

        return latents

    def _read_mask(self, mask_path: str, dest_size=(64, 64)):
        org_mask = Image.open(mask_path).convert("L")
        mask = org_mask.resize(dest_size, Image.NEAREST)
        mask = np.array(mask) / 255
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = mask[np.newaxis, np.newaxis, ...]
        mask = torch.from_numpy(mask).half().to(self.args.device)

        return mask, org_mask


if __name__ == "__main__":
    bld = BlendedLatnetDiffusion()
    results = bld.edit_image(
        bld.args.init_image,
        bld.args.mask,
        prompts=[bld.args.prompt] * bld.args.batch_size,
        blending_percentage=bld.args.blending_start_percentage,
    )
    results_flat = np.concatenate(results, axis=1)
    Image.fromarray(results_flat).save(bld.args.output_path)
