from flask import Flask, request, jsonify, Response, render_template

import io
import os
import logging
import numpy as np
from PIL import Image

import torch
import cv2
from omegaconf import OmegaConf

from diffusers import DDIMScheduler, UniPCMultistepScheduler
from diffusers.models import UNet2DConditionModel
from ref_encoder.latent_controlnet import ControlNetModel
from ref_encoder.adapter import *
from ref_encoder.reference_unet import ref_unet
from utils.pipeline import StableHairPipeline
from utils.pipeline_cn import StableDiffusionControlNetPipeline

# from hair_swap import HairFast, get_parser

# model_args = get_parser()
# hair_fast = HairFast(model_args.parse_args([]))
# model_parser = get_parser()
# model_args, _ = model_parser.parse_known_args()
# hair_fast = HairFast(model_args)

class StableHair:
    def __init__(self, config="./configs/hair_transfer.yaml", device="cuda", weight_dtype=torch.float32) -> None:
        print("Initializing Stable Hair Pipeline...")
        self.config = OmegaConf.load(config)
        self.device = device

        ### Load vae controlnet
        unet = UNet2DConditionModel.from_pretrained(self.config.pretrained_model_path, subfolder="unet").to(device)
        controlnet = ControlNetModel.from_unet(unet).to(device)
        _state_dict = torch.load(os.path.join(self.config.pretrained_folder, self.config.controlnet_path))
        controlnet.load_state_dict(_state_dict, strict=False)
        controlnet.to(weight_dtype)

        ### >>> create pipeline >>> ###
        self.pipeline = StableHairPipeline.from_pretrained(
            self.config.pretrained_model_path,
            controlnet=controlnet,
            safety_checker=None,
            torch_dtype=weight_dtype,
        ).to(device)
        self.pipeline.scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config)

        ### load Hair encoder/adapter
        self.hair_encoder = ref_unet.from_pretrained(self.config.pretrained_model_path, subfolder="unet").to(device)
        _state_dict = torch.load(os.path.join(self.config.pretrained_folder, self.config.encoder_path))
        self.hair_encoder.load_state_dict(_state_dict, strict=False)
        self.hair_adapter = adapter_injection(self.pipeline.unet, device=self.device, dtype=torch.float16, use_resampler=False)
        _state_dict = torch.load(os.path.join(self.config.pretrained_folder, self.config.adapter_path))
        self.hair_adapter.load_state_dict(_state_dict, strict=False)

        ### load bald converter
        bald_converter = ControlNetModel.from_unet(unet).to(device)
        _state_dict = torch.load(self.config.bald_converter_path)
        bald_converter.load_state_dict(_state_dict, strict=False)
        bald_converter.to(dtype=weight_dtype)
        del unet

        ### create pipeline for hair removal
        self.remove_hair_pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            self.config.pretrained_model_path,
            controlnet=bald_converter,
            safety_checker=None,
            torch_dtype=weight_dtype,
        )
        self.remove_hair_pipeline.scheduler = UniPCMultistepScheduler.from_config(self.remove_hair_pipeline.scheduler.config)
        self.remove_hair_pipeline = self.remove_hair_pipeline.to(device)

        ### move to fp16
        self.hair_encoder.to(weight_dtype)
        self.hair_adapter.to(weight_dtype)

        print("Initialization Done!")

    def Hair_Transfer(self, source_image, reference_image, random_seed, step, guidance_scale, scale, controlnet_conditioning_scale):
        prompt = ""
        n_prompt = ""
        random_seed = int(random_seed)
        step = int(step)
        guidance_scale = float(guidance_scale)
        scale = float(scale)
        controlnet_conditioning_scale = float(controlnet_conditioning_scale)

        # load imgs
        H, W, C = source_image.shape

        # generate images
        set_scale(self.pipeline.unet, scale)
        generator = torch.Generator(device="cuda")
        generator.manual_seed(random_seed)
        sample = self.pipeline(
            prompt,
            negative_prompt=n_prompt,
            num_inference_steps=step,
            guidance_scale=guidance_scale,
            width=W,
            height=H,
            controlnet_condition=source_image,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            generator=generator,
            reference_encoder=self.hair_encoder,
            ref_image=reference_image,
        ).samples
        return sample, source_image, reference_image

    def get_bald(self, id_image, scale):
        H, W = id_image.size
        scale = float(scale)
        image = self.remove_hair_pipeline(
            prompt="",
            negative_prompt="",
            num_inference_steps=30,
            guidance_scale=1.5,
            width=W,
            height=H,
            image=id_image,
            controlnet_conditioning_scale=scale,
            generator=None,
        ).images[0]

        return image


model = StableHair(config="./configs/hair_transfer.yaml", weight_dtype=torch.float32)


def resize_image(image, target_size=(1024, 1024)):
    """Resize the image to the target size (e.g., 1024x1024)."""
    image = image.resize(target_size, Image.LANCZOS)
    return image

def ensure_rgb(image):
    """Ensure the image has 3 channels (RGB). If the image has an alpha channel (RGBA), it will be converted to RGB."""
    if image.mode == 'RGBA':  # Check if the image has an alpha channel
        image = image.convert('RGB')  # Remove alpha channel and convert to RGB
    return image

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/wig_stick', methods=['POST'])
def wig_stick():
    # Check if the post request has the files
    if 'face' not in request.files or 'shape' not in request.files:
        return jsonify({"message": "Missing one or more required files (face, shape, color)"}), 400

    # Get the files from the request
    face_file = request.files['face']
    shape_file = request.files['shape']

    # If color is part of the input, handle it
    color_file = shape_file  # Using shape as color if color is not provided

    # Check if files are valid (you can add other checks here like file extensions)
    if face_file.filename == '' or shape_file.filename == '':
        return jsonify({"message": "No selected file"}), 400

    try:
        # Log the file size and content type
        logging.info(f"Received face file: {face_file.filename}, MIME type: {face_file.content_type}, size: {len(face_file.read())} bytes")
        face_file.stream.seek(0)  # Reset the pointer after logging the file size

        logging.info(f"Received shape file: {shape_file.filename}, MIME type: {shape_file.content_type}, size: {len(shape_file.read())} bytes")
        shape_file.stream.seek(0)  # Reset the pointer

        if color_file:
            logging.info(f"Received color file: {color_file.filename}, MIME type: {color_file.content_type}, size: {len(color_file.read())} bytes")
            color_file.stream.seek(0)  # Reset the pointer

        # Try opening the files
        face_image = Image.open(io.BytesIO(face_file.read()))
        shape_image = Image.open(io.BytesIO(shape_file.read()))
        color_image = shape_image

        # Resize all images to a consistent size (1024x1024)
        target_size = (1024, 1024)  # Resize to a fixed size like 1024x1024
        face_image = resize_image(ensure_rgb(face_image), target_size)
        shape_image = resize_image(ensure_rgb(shape_image), target_size)
        if color_file:
            color_image = resize_image(ensure_rgb(color_image), target_size)

        # Log the resized image dimensions
        logging.info(f"Resized Face image size: {face_image.size}, Shape image size: {shape_image.size}")
        if color_file:
            logging.info(f"Resized Color image size: {color_image.size}")

        # Process the images here (e.g., hair swapping)
        # final_tensor = hair_fast.swap(face_image, shape_image, color_image)
        final_tensor = hair_fast.swap(face_image, shape_image, color_image, align=True)

        # Convert tensor to PIL Image
        final_image = final_tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        final_image = (final_image * 255).astype(np.uint8)
        final_pil_image = Image.fromarray(final_image)

        # Convert the output image to a byte stream
        img_byte_arr = io.BytesIO()
        final_pil_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        # Return the final image as a response
        return Response(img_byte_arr, mimetype='image/png')

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return jsonify({"message": "Error in hair swapping", "error": str(e)}), 500
    
@app.route('/get_wig', methods=['POST'])
def get_wig():
    # Only check for 'face' since shape is loaded from path
    if 'face' not in request.files:
        return jsonify({"message": "Missing required file: face"}), 400

    face_file = request.files['face']
    choose_file = request.form.get('shape', '1')
    align = request.form.get('align', '1')
    align = align=='1'
    files = os.listdir("static/wig")
    if face_file.filename == '' or f"{choose_file}.png" not in files:
        return jsonify({"message": "No selected face file"}), 400

    try:
        # Open input images
        face_image = Image.open(io.BytesIO(face_file.read()))
        shape_image = Image.open(f"static/wig/{choose_file}.png")

        # Resize images
        target_size = (1024, 1024)
        face_image = resize_image(ensure_rgb(face_image), target_size)
        shape_image = resize_image(ensure_rgb(shape_image), target_size)

        face_image = ensure_rgb(face_image).resize((512, 512))
        ref_hair = ensure_rgb(shape_image).resize((512, 512))

        # Call your hair swap function
        id_image_np = np.array(face_image).astype('uint8')
        ref_hair_np = np.array(ref_hair).astype('uint8')

        # Convert back to PIL to apply model pipeline
        id_image = Image.fromarray(id_image_np, 'RGB')
        ref_hair = Image.fromarray(ref_hair_np, 'RGB')

        # Step 1: Get bald image
        id_image_bald = model.get_bald(id_image, 1)

        # Step 2: Convert inputs to numpy arrays
        id_image_bald = np.array(id_image_bald)
        ref_hair = np.array(ref_hair)

        # Step 3: Hair Transfer
        image, source_image, reference_image = model.Hair_Transfer(
            source_image=id_image_bald,
            reference_image=ref_hair,
            random_seed=-1,
            step=30,
            guidance_scale=1.5,
            scale=1.0,
            controlnet_conditioning_scale=1
        )
        final_pil_image = Image.fromarray((image * 255).astype(np.uint8))

        # Prepare response
        img_byte_arr = io.BytesIO()
        final_pil_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        return Response(img_byte_arr, mimetype='image/png')

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return jsonify({"message": "Error in hair swapping", "error": str(e)}), 500
    
@app.route('/web', methods=['GET'])
def web():
    wig_dir = 'static/wig'
    files = [os.path.splitext(f)[0] for f in os.listdir(wig_dir) if f.lower().endswith(('.png'))]
    return render_template('web.html', wig_files=files)

@app.route('/temp', methods=['GET'])
def temp():
    wig_dir = 'static/wig'
    files = [os.path.splitext(f)[0] for f in os.listdir(wig_dir) if f.lower().endswith(('.png'))]
    return render_template('temp.html', wig_files=files)

@app.route('/camera', methods=['GET'])
def camera():
    wig_dir = 'static/wig'
    files = [os.path.splitext(f)[0] for f in os.listdir(wig_dir) if f.lower().endswith(('.png'))]
    return render_template('camera.html', wig_files=files)

@app.route('/get_wig_list', methods=['GET'])
def get_wig_list():
    wig_dir = 'static/wig'
    domain = request.host_url.rstrip('/')
    files = [dict(id=os.path.splitext(f)[0], url=f"{domain}/{wig_dir}/{f}") for f in os.listdir(wig_dir) if f.lower().endswith(('.png'))]
    return files

@app.route('/test_image', methods=['POST'])
def test_image():
    face_file = request.files['face']
    choose_file = request.form.get('shape', '1')
    shape_image = Image.open(f"static/wig/{choose_file}.png")
    face_image = Image.open(io.BytesIO(face_file.read()))
    target_size = (1024, 1024)
    face_image = resize_image(ensure_rgb(face_image), target_size)
    shape_image = resize_image(ensure_rgb(shape_image), target_size)
    final_pil_image = face_image  # Just return wig

    img_byte_arr = io.BytesIO()
    final_pil_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return Response(img_byte_arr, mimetype='image/png')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)