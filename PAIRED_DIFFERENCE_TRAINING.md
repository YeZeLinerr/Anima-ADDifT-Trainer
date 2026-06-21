# Paired Image Difference LoRA Training

This mode trains an Anima LoRA from aligned before/after image pairs. It is
intended for localized transformations such as adding tattoos.

## Dataset layout

Use two directories with matching base filenames:

```text
tattoo/
  person_001.png
  person_002.png
clean/
  person_001.png
  person_002.png
```

The target image in `image_dir` contains the tattoo. The paired image in
`conditioning_data_dir` is the clean/no-tattoo reference. Each pair must have
the same dimensions and alignment. Captions belong beside the target images.

```toml
[general]
enable_bucket = true
bucket_no_upscale = true
min_bucket_reso = 512
max_bucket_reso = 1536
bucket_reso_steps = 64

[[datasets]]
resolution = [1024, 1024]
batch_size = 1
caption_extension = ".txt"

  [[datasets.subsets]]
  image_dir = "D:/dataset/tattoo"
  conditioning_data_dir = "D:/dataset/clean"
  num_repeats = 10
  flip_aug = false
  caption_prefix = "tattoo_style, "
```

Add these training arguments:

```toml
[training_arguments]
paired_difference_mode = true
paired_slider_scale = 0.25
paired_min_timestep = 500
paired_max_timestep = 1000
paired_difference_mask = true
paired_mask_normalize = true
paired_mask_threshold = 1.0
paired_background_weight = 0.1
cache_latents_to_disk = false
gradient_checkpointing = true
max_data_loader_n_workers = 0
persistent_data_loader_workers = false

[network_arguments]
network_module = "networks.lora_anima"
network_train_unet_only = true
```

Training uses an ADDifT-style prediction-matching objective. Both images receive
the same noise and timestep. In one direction, the clean reference with a
positive LoRA multiplier is trained to match the target prediction with LoRA
disabled. In the other direction, the target image with a negative multiplier
is trained to match the clean reference prediction with LoRA disabled. Thus
`+1` applies the effect stored in `image_dir`, while `-1` removes it. The
directions alternate while keeping only one DiT autograd graph in VRAM.

This differs from ordinary denoising LoRA training: the loss directly compares
the paired model predictions, forcing the LoRA to learn the transformation
between the aligned images.

Use `paired_min_timestep = 500` and `paired_max_timestep = 1000` for poses,
local decorations, tattoos, and other structural changes. For color, brightness,
or art-style changes, start with `200` and `400`.

Keep `paired_mask_normalize = true` for small localized edits. It normalizes
the soft mask by its effective area before averaging the full latent loss, so
a small tattoo is not weakened merely because it occupies few latent pixels.

For best results, avoid pose, lighting, crop, expression, clothing, or
background changes within a pair. Those changes will otherwise be learned as
part of the tattoo effect.
