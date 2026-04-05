import glob
import os
import kornia
import torch
from tqdm import tqdm
import torchvision
from diffusers import DDIMScheduler
from iif.component.task.single_view_prediction.pipeline_rgb2x import StableDiffusionAOVMatEstPipeline
from iif.task.task import Task
from iif.utils.image_io import load_ldr_image, show_image
from iif.utils.logging import init_logger


class SingleViewPrediction(Task):
    """
    A task for single-view prediction in the PIR framework.
    
    This task is designed to handle single-view prediction scenarios, where the model predicts outputs based on a single input view.
    """
    TASK_NAME = "1_single_view_prediction/rgbx"
    AOV_PROMPTS = {
        "albedo": "Albedo (diffuse basecolor)",
        "normal": "Camera-space Normal",
        "roughness": "Roughness",
        "metallic": "Metallicness", 
        "shading": "Irradiance (diffuse lighting)",
    }

    def __init__(self,
                 input,
                 output,
                 model,
                 sampling,
                 logging,
                 **kwargs):
        super().__init__()
        
        self.input = input
        self.output = output
        self.model = model
        self.sampling = sampling
        self.logging = logging

        self.module_logger = init_logger()

    def log_config(self, cfg):
        """Log the configuration of the single-view prediction task."""
        # Implement logging logic here
        pass

    def run(self):
        """Run the single-view prediction task."""
        # Prepare the model
        pipe = self.prepare_model()

        # Iterate over the input data
        files = self.input_iter()
        if self.logging["progress_bar"]:
            files = tqdm(files, desc="Processing files", unit="file")

        for idx, file_path in enumerate(files):
            # Predict
            self.run_prediction(pipe, file_path)

    def prepare_model(self):
        """Prepare the model for single-view prediction."""
        # Implement model preparation logic here
        # Load pipeline
        pipe = StableDiffusionAOVMatEstPipeline.from_pretrained(
            self.model["pretrained_model_name_or_path"],
            torch_dtype=torch.float16,
            cache_dir=self.model["cache_dir"],
        )
        pipe.scheduler = DDIMScheduler.from_config(
            pipe.scheduler.config, rescale_betas_zero_snr=True, timestep_spacing="trailing"
        )
        pipe.set_progress_bar_config(disable=True)
        pipe.to("cuda")

        pipe.enable_xformers_memory_efficient_attention()
        # pipe.enable_attention_slicing()

        return pipe
    
    def input_iter(self):
        """Iterate over the input data for single-view prediction."""
        # Implement input iteration logic here
        self.module_logger.info(f"Searching for input files in: {self.input['folder_path']}")
        for file_path in sorted(glob.glob(self.input["folder_path"] + '/**/*.png', recursive=True) 
                                + glob.glob(self.input["folder_path"] + '/**.JPG', recursive=True)):
            yield file_path

    @torch.no_grad()
    def run_prediction(self, pipe, file_path):
        """Run prediction on a single input file."""
        for aov_name in self.sampling["aovs"]:
            # Check if the AOV needs to be predicted
            output_path = self.get_output_path(file_path, aov_name, sample_id=self.sampling["num_samples_per_image"] - 1)
            if os.path.exists(output_path):
                self.module_logger.info(f"Output already exists: {output_path}, skipping prediction.")
                continue
            else:
                # Create a place-holder
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, "w") as f:
                    f.write("In progress!")

            # Load the image
            image = load_ldr_image(file_path, from_srgb=True)
            image = torch.from_numpy(image).permute(2, 0, 1).to("cuda") 
            
            # Run any transforms
            if self.input["apply_smoothing"]:
                image = kornia.filters.bilateral_blur(image[None], 9, 0.1, (3, 3))[0]

            # Preprocess the image
            photo = self.preprocess_image(image)

            # Iterate over the required number of samples
            aov_prompt = self.AOV_PROMPTS[aov_name]
            for sample_id in range(self.sampling["num_samples_per_image"]):
                # Create generator
                generator = torch.Generator(device="cuda").manual_seed(sample_id + self.sampling["seed"])

                # Generate the prefiction
                generated_image = pipe(
                    prompt=aov_prompt,
                    photo=photo,
                    num_inference_steps=self.sampling["num_inference_steps"],
                    height=photo.shape[1],
                    width=photo.shape[2],
                    generator=generator,
                    required_aovs=[aov_name],
                    do_gamma_correction=self.sampling["do_gamma_correction"],
                ).images[0][0]
                generated_image = torchvision.transforms.Resize((image.shape[1], image.shape[2]))(generated_image)

                # Save the output
                out_path = self.get_output_path(file_path, aov_name, sample_id)
                self.save_output(generated_image, out_path)

    def preprocess_image(self, image):
        """Preprocess the input image for prediction."""
        # Check if the width and height are multiples of 8. If not, crop it using torchvision.transforms.CenterCrop
        old_height = image.shape[1]
        old_width = image.shape[2]
        new_height = old_height
        new_width = old_width
        radio = old_height / old_width
        max_side = 1024
        if old_height > old_width:
            new_height = max_side
            new_width = int(new_height / radio)
        else:
            new_width = max_side
            new_height = int(new_width * radio)

        if new_width % 8 != 0 or new_height % 8 != 0:
            new_width = new_width // 8 * 8
            new_height = new_height // 8 * 8

        image = torchvision.transforms.Resize((new_height, new_width))(image)
        return image

    def get_output_path(self, file_path, aov, sample_id):
        """Get the output path for saving results."""
        out_path = os.path.join(self.output["folder_path"], os.path.relpath(file_path, self.input["folder_path"]))
        out_path_parent, out_path_basename = os.path.dirname(out_path), os.path.basename(out_path)
        if self.sampling["num_samples_per_image"] == 1:
            return os.path.join(out_path_parent, aov, out_path_basename.replace('.JPG', '.png'))
        return os.path.join(out_path_parent, aov, out_path_basename.replace('.JPG', '.png').replace('.png', f'_{sample_id:03d}.png'))

    def save_output(self, generated_image, out_path):
        """Save the output data from single-view prediction."""
        generated_image.save(out_path)
