from diffusers import DiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel
import torch
import os
import bayesian
import argparse
import socket
from safetensors.torch import load_file
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from oft_utils.attention_processor import OFTAttnProcessor

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--class_token",
        type=str,
        default=None,
        required=True,
        help="class_token",
    )
    parser.add_argument(
        "--lora",
        action="store_true",
        default=False,
        required=False,
        help="whether to use lora",
    )
    parser.add_argument(
        "--oft",
        action="store_true",
        default=False,
        required=False,
        help="whether to use oft",
    )
    parser.add_argument(
        "--is_live",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--gray_latent",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--zero_prompt",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--inference_step",
        type=int,
        default=50,
        required=False,
        help="step for inference",
    )
    bayesian.add_bayesian_parser(parser)
    args = parser.parse_args()
    # "/root/autodl-fs/log_db_bayes/backpack/checkpoint-{step}"
    model_path = args.model_path
    # "./log_db_bayes/backpacksrc/"
    output_path = args.output_path

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if model_path == 'original':
        pipe = DiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", dtype=torch.float16, safety_checker=None,
            cache_dir='/root/.cache/huggingface/hub', local_files_only=True)
    else:
        if 'autodl' in socket.gethostname():
            pipe = DiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5", dtype=torch.float16, safety_checker=None,
                cache_dir='/root/.cache/huggingface/hub', local_files_only=True)
        else:
            pipe = DiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5", dtype=torch.float16, safety_checker=None, cache_dir='./hub',
                local_files_only=True)
        pipe.unet = bayesian.convert_with_config(pipe.unet, args)

        if args.lora:
            pipe.load_lora_weights(f'{args.model_path}/model.safetensors')
        elif args.oft:
            oft_params = load_file(f'{args.model_path}/model.safetensors')
            oft_attn_procs = {}
            for name in pipe.unet.attn_processors.keys():
                cross_attention_dim = None if name.endswith("attn1.processor") else pipe.unet.config.cross_attention_dim
                if name.startswith("mid_block"):
                    hidden_size = pipe.unet.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(reversed(pipe.unet.config.block_out_channels))[block_id]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = pipe.unet.config.block_out_channels[block_id]
                else:
                    raise NotImplementedError

                oft_attn_procs[name] = OFTAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim,
                                                        eps=6e-5, r=4, is_coft=False)
                for param_key, param_value in oft_attn_procs[name].named_parameters():
                    if name + '.' + param_key in oft_params:
                        param_value.data.copy_(oft_params[name + '.' + param_key])
                    else:
                        param_value.data.copy_(oft_params[name + '.' + param_key[:-1] + 'mu_' + param_key[-1]])
                pass

            pipe.unet.set_attn_processor(oft_attn_procs)

        else:
            pipe.unet.load_state_dict(load_file(f'{args.model_path}/unet/diffusion_pytorch_model.safetensors'))

    if args.test_sigma >= 0.0:
        bayesian.reset_sigma(pipe.unet, args.test_sigma)
    pipe = pipe.to("cuda")
    images = []

    unique_token = "qwe"
    # class_token = "backpack"
    class_token = args.class_token
    is_not_live = True
    if ('cat' in class_token) or ('dog' in class_token):
        is_not_live = False

    if is_not_live and not args.is_live:
        prompt_list = [
            f"a {unique_token} {class_token} in the jungle",
            f"a {unique_token} {class_token} in the snow",
            f"a {unique_token} {class_token} on the beach",
            f"a {unique_token} {class_token} on a cobblestone street",
            f"a {unique_token} {class_token} on top of pink fabric",
            f"a {unique_token} {class_token} on top of a wooden floor",
            f"a {unique_token} {class_token} with a city in the background",
            f"a {unique_token} {class_token} with a mountain in the background",
            f"a {unique_token} {class_token} with a blue house in the background",
            f"a {unique_token} {class_token} on top of a purple rug in a forest",
            f"a {unique_token} {class_token} with a wheat field in the background",
            f"a {unique_token} {class_token} with a tree and autumn leaves in the background",
            f"a {unique_token} {class_token} with the Eiffel Tower in the background",
            f"a {unique_token} {class_token} floating on top of water",
            f"a {unique_token} {class_token} floating in an ocean of milk",
            f"a {unique_token} {class_token} on top of green grass with sunflowers around it",
            f"a {unique_token} {class_token} on top of a mirror",
            f"a {unique_token} {class_token} on top of the sidewalk in a crowded street",
            f"a {unique_token} {class_token} on top of a dirt road",
            f"a {unique_token} {class_token} on top of a white rug",
            f"a red {unique_token} {class_token}",
            f"a purple {unique_token} {class_token}",
            f"a shiny {unique_token} {class_token}",
            f"a wet {unique_token} {class_token}",
            f"a cube shaped {unique_token} {class_token}"
        ]
    else:
        prompt_list = (
            f"a {unique_token} {class_token} in the jungle",
            f"a {unique_token} {class_token} in the snow",
            f"a {unique_token} {class_token} on the beach",
            f"a {unique_token} {class_token} on a cobblestone street",
            f"a {unique_token} {class_token} on top of pink fabric",
            f"a {unique_token} {class_token} on top of a wooden floor",
            f"a {unique_token} {class_token} with a city in the background",
            f"a {unique_token} {class_token} with a mountain in the background",
            f"a {unique_token} {class_token} with a blue house in the background",
            f"a {unique_token} {class_token} on top of a purple rug in a forest",
            f"a {unique_token} {class_token} wearing a red hat",
            f"a {unique_token} {class_token} wearing a santa hat",
            f"a {unique_token} {class_token} wearing a rainbow scarf",
            f"a {unique_token} {class_token} wearing a black top hat and a monocle",
            f"a {unique_token} {class_token} in a chef outfit",
            f"a {unique_token} {class_token} in a firefighter outfit",
            f"a {unique_token} {class_token} in a police outfit",
            f"a {unique_token} {class_token} wearing pink glasses",
            f"a {unique_token} {class_token} wearing a yellow shirt",
            f"a {unique_token} {class_token} in a purple wizard outfit",
            f"a red {unique_token} {class_token}",
            f"a purple {unique_token} {class_token}",
            f"a shiny {unique_token} {class_token}",
            f"a wet {unique_token} {class_token}",
            f"a cube shaped {unique_token} {class_token}"
        )

    for prompt in prompt_list:
        if args.gray_latent is not None:
            latent = torch.ones([4, 4, 64, 64]).cuda() * args.gray_latent
        else:
            latent = None
        if args.zero_prompt:
            images = images + pipe(prompt="a qwe backpack", latents=latent, num_inference_steps=args.inference_step, guidance_scale=7.5, num_images_per_prompt=4).images
        else:
            images = images + pipe(prompt=prompt, latents=latent, num_inference_steps=args.inference_step, guidance_scale=7.5,
                                   num_images_per_prompt=4).images
        if args.inference_step >= 500:
            break
    for i, im in enumerate(images):
        im.save(f"{output_path}/{i}.png")


if __name__ == "__main__":
    main()
