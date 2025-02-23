from PIL import Image
import logging
from idm_vton.src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from idm_vton.src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from idm_vton.src.unet_hacked_tryon import UNet2DConditionModel
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTextModelWithProjection,
)
from diffusers import DDPMScheduler,AutoencoderKL
from typing import List
import argparse
import torch
from transformers import AutoTokenizer
from idm_vton.utils_mask import get_mask_location
from torchvision import transforms
from idm_vton import apply_net
from idm_vton.preprocess.humanparsing.run_parsing import Parsing
from idm_vton.preprocess.openpose.run_openpose import OpenPose
from idm_vton.detectron2.data.detection_utils import convert_PIL_to_numpy,_apply_exif_orientation
from torchvision.transforms.functional import to_pil_image

logger = logging.getLogger(__name__)  
logger.setLevel(logging.WARN)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARN)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

def parse_args(device, denoise_steps):
    Float_device = torch.float32
    if device == "cuda":
        Float_device = torch.float16

    base_path = 'yisol/IDM-VTON'

    noise_scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(
        base_path,                         
        subfolder="vae",
        torch_dtype=Float_device,
    )
    unet = UNet2DConditionModel.from_pretrained(
        base_path,
        subfolder="unet",
        torch_dtype=Float_device,
    )
    unet_encoder = UNet2DConditionModel_ref.from_pretrained(
        base_path,
        subfolder="unet_encoder",
        torch_dtype=Float_device,
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        base_path,
        subfolder="image_encoder",
        torch_dtype=Float_device,
    )
    text_encoder_one = CLIPTextModel.from_pretrained(
        base_path,
        subfolder="text_encoder",
        torch_dtype=Float_device,
    )
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        base_path,
        subfolder="text_encoder_2",
        torch_dtype=Float_device,
    )
    tokenizer_one = AutoTokenizer.from_pretrained(
        base_path,
        subfolder="tokenizer",
        revision=None,
        use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        base_path,
        subfolder="tokenizer_2",
        revision=None,
        use_fast=False,
    )

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    unet_encoder.requires_grad_(False)
    image_encoder.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    parsing_model = Parsing(0)
    openpose_model = OpenPose(0)

    tensor_transfrom = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    pipe = TryonPipeline.from_pretrained(
            base_path,
            unet=unet,
            vae=vae,
            feature_extractor= CLIPImageProcessor(),
            text_encoder = text_encoder_one,
            text_encoder_2 = text_encoder_two,
            tokenizer = tokenizer_one,
            tokenizer_2 = tokenizer_two,
            scheduler = noise_scheduler,
            image_encoder=image_encoder,
            torch_dtype=Float_device,
    )
    pipe.unet_encoder = unet_encoder

    parser = argparse.ArgumentParser(description="implementation of IDM-VTON")
    parser.add_argument("--openpose_model",default=openpose_model,)
    parser.add_argument("--pipe",default=pipe,)
    parser.add_argument("--parsing_model",default=parsing_model,)
    parser.add_argument("--tensor_transfrom",default=tensor_transfrom,)
    parser.add_argument("--garment_des",default="",)
    parser.add_argument("--Float_device",default=Float_device,)
    parser.add_argument("--denoise_steps",default=denoise_steps,)
    parser.add_argument("--seed",default=42,)
    
    return parser.parse_args()

def try_on(human, garm_img, category, device, args):
    
    args.openpose_model.preprocessor.body_estimation.model.to(device)
    args.pipe.to(device)
    args.pipe.unet_encoder.to(device)

    garm_img= garm_img.convert("RGB").resize((768,1024))
    human_img_orig = human.convert("RGB")    
    human_img = human_img_orig.resize((768,1024))
    
    keypoints = args.openpose_model(human_img.resize((384,512)), device)
    model_parse, _ = args.parsing_model(human_img.resize((384,512)))
    mask, mask_gray = get_mask_location('hd', category, model_parse, keypoints)
    mask = mask.resize((768,1024))
    mask_gray = (1-transforms.ToTensor()(mask)) * args.tensor_transfrom(human_img)
    mask_gray = to_pil_image((mask_gray+1.0)/2.0)

    human_img_arg = _apply_exif_orientation(human_img.resize((384,512)))
    human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")

    # temp_args = apply_net.create_argument_parser().parse_args(('show', './configs/densepose_rcnn_R_50_FPN_s1x.yaml', './ckpt/densepose/model_final_162be9.pkl', 'dp_segm', '-v', '--opts', 'MODEL.device', 'cuda'))
    temp_args = apply_net.create_argument_parser().parse_args(('show', './idm_vton/configs/densepose_rcnn_R_50_FPN_s1x.yaml', './idm_vton/ckpt/densepose/model_final_162be9.pkl', 'dp_segm', '-v', '--opts', 'MODEL.DEVICE', device))
    # verbosity = getattr(temp_args, "verbosity", None)

    pose_img = temp_args.func(temp_args,human_img_arg)    
    pose_img = pose_img[:,:,::-1]    
    pose_img = Image.fromarray(pose_img).resize((768,1024))
    
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                prompt = "model is wearing " + args.garment_des
                negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
                with torch.inference_mode():
                    (
                        prompt_embeds,
                        negative_prompt_embeds,
                        pooled_prompt_embeds,
                        negative_pooled_prompt_embeds,
                    ) = args.pipe.encode_prompt(
                        prompt,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=True,
                        negative_prompt=negative_prompt,
                    )
                    prompt = "a photo of " + args.garment_des
                    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
                    if not isinstance(prompt, List):
                        prompt = [prompt] * 1
                    if not isinstance(negative_prompt, List):
                        negative_prompt = [negative_prompt] * 1
                    with torch.inference_mode():
                        (
                            prompt_embeds_c,
                            _,
                            _,
                            _,
                        ) = args.pipe.encode_prompt(
                            prompt,
                            num_images_per_prompt=1,
                            do_classifier_free_guidance=False,
                            negative_prompt=negative_prompt,
                        )

                    pose_img =  args.tensor_transfrom(pose_img).unsqueeze(0).to(device,args.Float_device)
                    garm_tensor =  args.tensor_transfrom(garm_img).unsqueeze(0).to(device,args.Float_device)
                    generator = torch.Generator(device).manual_seed(args.seed) if args.seed is not None else None
                    print("Generating...:")
                    output = args.pipe(
                        prompt_embeds=prompt_embeds.to(device,args.Float_device),
                        negative_prompt_embeds=negative_prompt_embeds.to(device,args.Float_device),
                        pooled_prompt_embeds=pooled_prompt_embeds.to(device,args.Float_device),
                        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device,args.Float_device),
                        num_inference_steps=args.denoise_steps,
                        generator=generator,
                        strength = 1.0,
                        pose_img = pose_img.to(device,args.Float_device),
                        text_embeds_cloth=prompt_embeds_c.to(device,args.Float_device),
                        cloth = garm_tensor.to(device,args.Float_device),
                        mask_image=mask,
                        image=human_img, 
                        height=1024,
                        width=768,
                        ip_adapter_image = garm_img.resize((768,1024)),
                        guidance_scale=2.0,
                        add_watermark = True,
                    )[0]

    return output[0]

def generate(human, garm_imgs, categories, device, seed=30):
    """
    generate virtual try-on by IDM-VTON
    human = PIL object: human picture
    garm_imgs = list of PIL object: garment images to try-on
    categories = list of garment's category (big category)
    device = cuda or cpu
    """
    label = {
        "Dress": "dresses",
        "Top": "upper_body",
        "Bottom": "lower_body"
    }
    try:
        args = parse_args(device, seed)
        output = try_on(human, garm_imgs[0], label[categories[0]], device, args)
        if categories[0] != "Dress":
            output = try_on(output, garm_imgs[1], label[categories[1]], device, args)
    except torch.cuda.OutOfMemoryError as e:
        logger.warning("VRAM of the gpu is not enough. Continue generating the virtual try-on using cpu...")
        args = parse_args("cpu", seed)
        output = try_on(human, garm_imgs[0], label[categories[0]], "cpu", args)
        if categories[0] != "Dress":
            output = try_on(output, garm_imgs[1], label[categories[1]], "cpu", args)
    
    
    return output