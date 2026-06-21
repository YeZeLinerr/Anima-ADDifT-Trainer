# Anima LoRA training script

import argparse
import math
import os
from typing import Any, Optional, Union

import torch
from accelerate import Accelerator
from library.device_utils import init_ipex, clean_memory_on_device

init_ipex()

from library import anima_models, anima_train_utils, anima_utils, strategy_anima, strategy_base, train_util
import train_network
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


class AnimaNetworkTrainer(train_network.NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.sample_prompts_te_outputs = None
        self.vae = None
        self.vae_scale = None
        self.qwen3_text_encoder = None
        self.qwen3_tokenizer = None
        self.t5_tokenizer = None
        self.tokenize_strategy = None
        self.text_encoding_strategy = None
        self._paired_slider_next_positive = True
        self._paired_slider_multiplier_active = False

    def assert_extra_args(
        self,
        args,
        train_dataset_group: Union[train_util.DatasetGroup, train_util.MinimalDataset],
        val_dataset_group: Optional[train_util.DatasetGroup],
    ):
        if args.cache_text_encoder_outputs_to_disk and not args.cache_text_encoder_outputs:
            logger.warning(
                "cache_text_encoder_outputs_to_disk is enabled, so cache_text_encoder_outputs is also enabled"
            )
            args.cache_text_encoder_outputs = True

        global_dropout_rate = getattr(args, 'caption_dropout_rate', 0.0)
        max_subset_dropout = 0.0
        if hasattr(train_dataset_group, 'datasets'):
            for dataset in train_dataset_group.datasets:
                for subset in dataset.subsets:
                    if subset.caption_dropout_rate > 0:
                        max_subset_dropout = max(max_subset_dropout, subset.caption_dropout_rate)
                        subset.caption_dropout_rate = 0.0

        if max_subset_dropout > 0 and global_dropout_rate == 0:
            logger.info(f"Migrating subset caption dropout rate ({max_subset_dropout}) to global level for Anima strategy")
            args.caption_dropout_rate = max_subset_dropout
        elif global_dropout_rate > 0:
            logger.info(f"Using global embedding-level caption dropout rate: {global_dropout_rate}")

        if args.cache_text_encoder_outputs:
            assert (
                train_dataset_group.is_text_encoder_output_cacheable()
            ), "when caching Text Encoder output, shuffle_caption, token_warmup_step or caption_tag_dropout_rate cannot be used"

        assert (
            args.network_train_unet_only or not args.cache_text_encoder_outputs
        ), "network for Text Encoder cannot be trained with caching Text Encoder outputs"

        assert (
            args.blocks_to_swap is None or args.blocks_to_swap == 0
        ) or not args.cpu_offload_checkpointing, "blocks_to_swap is not supported with cpu_offload_checkpointing"

        if getattr(args, 'unsloth_offload_checkpointing', False):
            if not args.gradient_checkpointing:
                logger.warning("unsloth_offload_checkpointing is enabled, so gradient_checkpointing is also enabled")
                args.gradient_checkpointing = True
            assert not args.cpu_offload_checkpointing, \
                "Cannot use both --unsloth_offload_checkpointing and --cpu_offload_checkpointing"
            assert (
                args.blocks_to_swap is None or args.blocks_to_swap == 0
            ), "blocks_to_swap is not supported with unsloth_offload_checkpointing"

        # Attention: validate availability
        if getattr(args, 'flash_attn', False):
            try:
                if not anima_models.FLASH_ATTN_AVAILABLE:
                    raise ImportError("No supported Flash Attention backend is installed")
                logger.info(f"Flash Attention enabled for DiT blocks ({anima_models.FLASH_ATTN_BACKEND})")
            except ImportError:
                logger.warning("flash_attn package not installed, falling back to PyTorch SDPA")
                args.flash_attn = False
                
        if getattr(args, 'blockwise_fused_optimizers', False):
            raise ValueError("blockwise_fused_optimizers is not supported with LoRA/NetworkTrainer")

        if getattr(args, "paired_difference_mode", False):
            if not args.gradient_checkpointing:
                logger.warning("Enabling gradient_checkpointing for paired slider memory efficiency.")
                args.gradient_checkpointing = True
            if args.max_data_loader_n_workers != 0:
                logger.warning(
                    "Setting max_data_loader_n_workers=0 for paired slider training "
                    "to avoid large Windows worker-process memory duplication."
                )
                args.max_data_loader_n_workers = 0
            if args.persistent_data_loader_workers:
                logger.warning("Disabling persistent_data_loader_workers for paired slider training.")
                args.persistent_data_loader_workers = False
            if args.cache_latents:
                raise ValueError(
                    "paired_difference_mode requires cache_latents=false because both target and reference "
                    "images must be encoded by the VAE with identical transforms during training."
                )
            if not hasattr(train_dataset_group, "datasets") or not all(
                isinstance(dataset, train_util.ControlNetDataset) for dataset in train_dataset_group.datasets
            ):
                raise ValueError(
                    "paired_difference_mode requires conditioning_data_dir in every dataset subset. "
                    "Put tattoo images in image_dir and matching clean images in conditioning_data_dir."
                )
            if not args.network_train_unet_only:
                raise ValueError(
                    "paired_difference_mode currently requires network_train_unet_only=true. "
                    "The reference branch freezes the LoRA and is not intended to train the text encoder."
                )
            if not hasattr(self, "is_swapping_blocks"):
                self.is_swapping_blocks = False
            if not 0.0 <= args.paired_background_weight <= 1.0:
                raise ValueError("paired_background_weight must be between 0 and 1")
            if args.paired_slider_scale <= 0.0:
                raise ValueError("paired_slider_scale must be > 0")
            if not 0 <= args.paired_min_timestep < args.paired_max_timestep <= 1000:
                raise ValueError(
                    "paired timesteps must satisfy "
                    "0 <= paired_min_timestep < paired_max_timestep <= 1000"
                )
            if getattr(args, "ip_noise_gamma", None):
                raise ValueError("paired_difference_mode does not support ip_noise_gamma")
            if args.paired_mask_threshold <= 0.0:
                raise ValueError("paired_mask_threshold must be > 0")
            logger.info(
                "Paired Image Difference Mode enabled with ADDifT-style prediction matching: "
                "image_dir=target, conditioning_data_dir=reference, "
                f"timesteps={args.paired_min_timestep}-{args.paired_max_timestep}"
            )

        train_dataset_group.verify_bucket_reso_steps(8)  # WanVAE spatial downscale = 8
        if val_dataset_group is not None:
            val_dataset_group.verify_bucket_reso_steps(8)

    def load_target_model(self, args, weight_dtype, accelerator):
        # Load Qwen3 text encoder (tokenizers already loaded in get_tokenize_strategy)
        logger.info("Loading Qwen3 text encoder...")
        self.qwen3_text_encoder, _ = anima_utils.load_qwen3_text_encoder(
            args.qwen3_path, dtype=weight_dtype, device="cpu"
        )
        self.qwen3_text_encoder.eval()

        # Parse transformer_dtype
        transformer_dtype = None
        if hasattr(args, 'transformer_dtype') and args.transformer_dtype is not None:
            transformer_dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            transformer_dtype = transformer_dtype_map.get(args.transformer_dtype, None)

        # Load DiT
        logger.info("Loading Anima DiT...")
        
        dit = anima_utils.load_anima_dit(
            args.dit_path,
            dtype=weight_dtype,
            device="cpu",
            transformer_dtype=transformer_dtype,
            llm_adapter_path=getattr(args, 'llm_adapter_path', None),
            disable_mmap=getattr(args, 'disable_mmap_load_safetensors', False),
        )

        # Attention backend
        if getattr(args, 'flash_attn', False):
            dit.set_flash_attn(True)

        # Store unsloth preference so that when the base NetworkTrainer calls
        self._use_unsloth_offload_checkpointing = getattr(args, 'unsloth_offload_checkpointing', False)

        # Block swap
        self.is_swapping_blocks = args.blocks_to_swap is not None and args.blocks_to_swap > 0
        if self.is_swapping_blocks:
            logger.info(f"enable block swap: blocks_to_swap={args.blocks_to_swap}")
            dit.enable_block_swap(args.blocks_to_swap, accelerator.device)

        # Load VAE
        logger.info("Loading Anima VAE...")
        self.vae, vae_mean, vae_std, self.vae_scale = anima_utils.load_anima_vae(
            args.vae_path, dtype=weight_dtype, device="cpu"
        )

        # Return format: (model_type, text_encoders, vae, unet)
        return "anima", [self.qwen3_text_encoder], self.vae, dit

    def get_tokenize_strategy(self, args):
        # Load tokenizers from paths (called before load_target_model, so self.qwen3_tokenizer isn't set yet)
        self.tokenize_strategy = strategy_anima.AnimaTokenizeStrategy(
            qwen3_path=args.qwen3_path,
            t5_tokenizer_path=getattr(args, 't5_tokenizer_path', None),
            qwen3_max_length=args.qwen3_max_token_length,
            t5_max_length=args.t5_max_token_length,
        )
        # Store references so load_target_model can reuse them
        self.qwen3_tokenizer = self.tokenize_strategy.qwen3_tokenizer
        self.t5_tokenizer = self.tokenize_strategy.t5_tokenizer
        return self.tokenize_strategy

    def get_tokenizers(self, tokenize_strategy: strategy_anima.AnimaTokenizeStrategy):
        return [tokenize_strategy.qwen3_tokenizer]

    def get_latents_caching_strategy(self, args):
        return strategy_anima.AnimaLatentsCachingStrategy(
            args.cache_latents_to_disk, args.vae_batch_size, args.skip_cache_check
        )

    def get_text_encoding_strategy(self, args):
        caption_dropout_rate = getattr(args, 'caption_dropout_rate', 0.0)
        self.text_encoding_strategy = strategy_anima.AnimaTextEncodingStrategy(
            dropout_rate=caption_dropout_rate,
        )
        return self.text_encoding_strategy

    def post_process_network(self, args, accelerator, network, text_encoders, unet):
        pass

    def prepare_text_encoder_grad_ckpt_workaround(self, index, text_encoder):
        # Qwen3Model uses embed_tokens, not text_model.embeddings (CLIP-specific)
        text_encoder.embed_tokens.requires_grad_(True)

    def prepare_text_encoder_fp8(self, index, text_encoder, te_weight_dtype, weight_dtype):
        # Qwen3Model uses embed_tokens, not text_model.embeddings (CLIP-specific)
        text_encoder.embed_tokens.to(dtype=weight_dtype)

    def get_models_for_text_encoding(self, args, accelerator, text_encoders):
        if args.cache_text_encoder_outputs:
            return None  # no text encoders needed for encoding
        return text_encoders

    def get_text_encoders_train_flags(self, args, text_encoders):
        return [not args.network_train_unet_only]

    def is_train_text_encoder(self, args):
        return not args.network_train_unet_only

    def get_text_encoder_outputs_caching_strategy(self, args):
        if args.cache_text_encoder_outputs:
            return strategy_anima.AnimaTextEncoderOutputsCachingStrategy(
                args.cache_text_encoder_outputs_to_disk,
                args.text_encoder_batch_size,
                args.skip_cache_check,
                is_partial=False,
            )
        return None

    def cache_text_encoder_outputs_if_needed(
        self, args, accelerator: Accelerator, unet, vae, text_encoders, dataset: train_util.DatasetGroup, weight_dtype
    ):
        if args.cache_text_encoder_outputs:
            if not args.lowram:
                logger.info("move vae and unet to cpu to save memory")
                org_vae_device = next(vae.parameters()).device
                org_unet_device = unet.device
                vae.to("cpu")
                unet.to("cpu")
                clean_memory_on_device(accelerator.device)

            logger.info("move text encoder to gpu")
            text_encoders[0].to(accelerator.device, dtype=weight_dtype)

            with accelerator.autocast():
                dataset.new_cache_text_encoder_outputs(text_encoders, accelerator)

            # cache sample prompts
            if args.sample_prompts is not None:
                logger.info(f"cache Text Encoder outputs for sample prompts: {args.sample_prompts}")

                tokenize_strategy = strategy_base.TokenizeStrategy.get_strategy()
                text_encoding_strategy = strategy_base.TextEncodingStrategy.get_strategy()

                prompts = train_util.load_prompts(args.sample_prompts)
                sample_prompts_te_outputs = {}
                with accelerator.autocast(), torch.no_grad():
                    for prompt_dict in prompts:
                        for p in [prompt_dict.get("prompt", ""), prompt_dict.get("negative_prompt", "")]:
                            if p not in sample_prompts_te_outputs:
                                logger.info(f"  cache TE outputs for: {p}")
                                tokens_and_masks = tokenize_strategy.tokenize(p)
                                sample_prompts_te_outputs[p] = text_encoding_strategy.encode_tokens(
                                    tokenize_strategy,
                                    text_encoders,
                                    tokens_and_masks,
                                    enable_dropout=False,
                                )
                self.sample_prompts_te_outputs = sample_prompts_te_outputs

            # Pre-cache unconditional embeddings for caption dropout before text encoder is deleted
            caption_dropout_rate = getattr(args, 'caption_dropout_rate', 0.0)
            text_encoding_strategy_for_uncond = strategy_base.TextEncodingStrategy.get_strategy()
            if caption_dropout_rate > 0.0:
                tokenize_strategy_for_uncond = strategy_base.TokenizeStrategy.get_strategy()
                with accelerator.autocast():
                    text_encoding_strategy_for_uncond.cache_uncond_embeddings(tokenize_strategy_for_uncond, text_encoders)

            accelerator.wait_for_everyone()

            # move text encoder back to cpu
            logger.info("move text encoder back to cpu")
            text_encoders[0].to("cpu")
            clean_memory_on_device(accelerator.device)

            if not args.lowram:
                logger.info("move vae and unet back to original device")
                vae.to(org_vae_device)
                unet.to(org_unet_device)
        else:
            text_encoders[0].to(accelerator.device, dtype=weight_dtype)

    def sample_images(self, accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet):
        text_encoders = text_encoder if isinstance(text_encoder, list) else [text_encoder]  # compatibility
        te = self.get_models_for_text_encoding(args, accelerator, text_encoders)
        qwen3_te = te[0] if te is not None else None

        anima_train_utils.sample_images(
            accelerator, args, epoch, global_step, unet, vae, self.vae_scale,
            qwen3_te, self.tokenize_strategy, self.text_encoding_strategy,
            self.sample_prompts_te_outputs,
        )

    def get_noise_scheduler(self, args: argparse.Namespace, device: torch.device) -> Any:
        noise_scheduler = anima_train_utils.FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000, shift=args.discrete_flow_shift
        )
        return noise_scheduler

    def encode_images_to_latents(self, args, vae, images):
        # images are already [-1,1] from IMAGE_TRANSFORMS, add temporal dim
        images = images.unsqueeze(2)  # (B, C, 1, H, W)
        # Ensure scale tensors are on the same device as images
        vae_device = images.device
        scale = [s.to(vae_device) if isinstance(s, torch.Tensor) else s for s in self.vae_scale]
        return vae.encode(images, scale)

    def shift_scale_latents(self, args, latents):
        # Latents already normalized by vae.encode with scale
        return latents

    def get_noise_pred_and_target(
        self,
        args,
        accelerator,
        noise_scheduler,
        latents,
        batch,
        text_encoder_conds,
        unet,
        network,
        weight_dtype,
        train_unet,
        is_train=True,
    ):
        # Sample noise
        noise = torch.randn_like(latents)

        # Get noisy model input and timesteps
        noisy_model_input, timesteps, sigmas = anima_train_utils.get_noisy_model_input_and_timesteps(
            args, latents, noise, accelerator.device, weight_dtype
        )

        # Gradient checkpointing support
        if args.gradient_checkpointing:
            noisy_model_input.requires_grad_(True)
            for t in text_encoder_conds:
                if t is not None and t.dtype.is_floating_point:
                    t.requires_grad_(True)

        # Unpack text encoder conditions
        prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask = text_encoder_conds

        # Move to device
        prompt_embeds = prompt_embeds.to(accelerator.device, dtype=weight_dtype)
        attn_mask = attn_mask.to(accelerator.device)
        t5_input_ids = t5_input_ids.to(accelerator.device, dtype=torch.long)
        t5_attn_mask = t5_attn_mask.to(accelerator.device)

        # Create padding mask
        bs = latents.shape[0]
        h_latent = latents.shape[-2]
        w_latent = latents.shape[-1]
        padding_mask = torch.zeros(
            bs, 1, h_latent, w_latent,
            dtype=weight_dtype, device=accelerator.device
        )

        # Prepare block swap
        if self.is_swapping_blocks:
            accelerator.unwrap_model(unet).prepare_block_swap_before_forward()

        # Call model (LLM adapter runs inside forward for DDP gradient sync)
        with torch.set_grad_enabled(is_train), accelerator.autocast():
            model_pred = unet(
                noisy_model_input,
                timesteps,
                prompt_embeds,
                padding_mask=padding_mask,
                source_attention_mask=attn_mask,
                t5_input_ids=t5_input_ids,
                t5_attn_mask=t5_attn_mask,
            )

        # Rectified flow target: noise - latents
        target = noise - latents

        # Loss weighting
        weighting = anima_train_utils.compute_loss_weighting_for_anima(
            weighting_scheme=args.weighting_scheme, sigmas=sigmas
        )

        # Differential output preservation
        if "custom_attributes" in batch:
            diff_output_pr_indices = []
            for i, custom_attributes in enumerate(batch["custom_attributes"]):
                if "diff_output_preservation" in custom_attributes and custom_attributes["diff_output_preservation"]:
                    diff_output_pr_indices.append(i)

            if len(diff_output_pr_indices) > 0:
                network.set_multiplier(0.0)
                with torch.no_grad(), accelerator.autocast():
                    if self.is_swapping_blocks:
                        accelerator.unwrap_model(unet).prepare_block_swap_before_forward()
                    model_pred_prior = unet(
                        noisy_model_input[diff_output_pr_indices],
                        timesteps[diff_output_pr_indices],
                        prompt_embeds[diff_output_pr_indices],
                        padding_mask=padding_mask[diff_output_pr_indices],
                        source_attention_mask=attn_mask[diff_output_pr_indices],
                        t5_input_ids=t5_input_ids[diff_output_pr_indices],
                        t5_attn_mask=t5_attn_mask[diff_output_pr_indices],
                    )
                network.set_multiplier(1.0)

                target[diff_output_pr_indices] = model_pred_prior.to(target.dtype)

        return model_pred, target, timesteps, weighting

    def get_paired_difference_predictions(
        self,
        args,
        accelerator,
        target_latents,
        reference_latents,
        text_encoder_conds,
        unet,
        network,
        weight_dtype,
        is_train=True,
    ):
        """Run one ADDifT-style signed prediction-matching branch.

        Both images use the same noise and timestep. One image is evaluated
        with LoRA disabled to provide a frozen base-model prediction, while
        the paired image is evaluated with a signed LoRA multiplier and is
        trained to match that prediction. The orientation alternates so the
        learned LoRA can be used in both positive and negative directions.
        """
        if not hasattr(network, "set_multiplier"):
            raise ValueError("paired_difference_mode requires a LoRA network that supports set_multiplier()")
        noise = torch.randn_like(target_latents)
        # ADDifT relies on a task-specific noise range instead of the normal
        # logit-normal training distribution. Values are exposed as 0..1000
        # for consistency with diffusion UIs, then normalized to Anima's 0..1
        # rectified-flow timestep.
        timestep_min = args.paired_min_timestep / 1000.0
        timestep_max = args.paired_max_timestep / 1000.0
        timesteps = torch.empty(
            target_latents.shape[0], device=accelerator.device, dtype=torch.float32
        ).uniform_(timestep_min, timestep_max)
        t_expanded = timesteps.view(-1, *([1] * (target_latents.ndim - 1)))
        target_noisy = (1.0 - t_expanded) * target_latents + t_expanded * noise
        reference_noisy = (1.0 - t_expanded) * reference_latents + t_expanded * noise
        target_noisy = target_noisy.to(weight_dtype)
        reference_noisy = reference_noisy.to(weight_dtype)
        timesteps = timesteps.to(weight_dtype)
        sigmas = timesteps.view(-1, 1)

        prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask = text_encoder_conds
        prompt_embeds = prompt_embeds.to(accelerator.device, dtype=weight_dtype)
        attn_mask = attn_mask.to(accelerator.device)
        t5_input_ids = t5_input_ids.to(accelerator.device, dtype=torch.long)
        t5_attn_mask = t5_attn_mask.to(accelerator.device)

        bs, h_latent, w_latent = target_latents.shape[0], target_latents.shape[-2], target_latents.shape[-1]
        padding_mask = torch.zeros(
            bs, 1, h_latent, w_latent, dtype=weight_dtype, device=accelerator.device
        )

        positive_branch = self._paired_slider_next_positive
        if is_train:
            self._paired_slider_next_positive = not self._paired_slider_next_positive

        # Effect-oriented ADDifT polarity:
        #   positive: reference with +LoRA should match target without LoRA
        #   negative: target with -LoRA should match reference without LoRA
        #
        # This makes inference multiplier +1 apply image_dir's effect and -1
        # remove it. The opposite orientation makes +1 learn target->reference,
        # which is especially obvious when the target pair is a black image.
        model_input = reference_noisy if positive_branch else target_noisy
        base_input = target_noisy if positive_branch else reference_noisy
        signed_scale = args.paired_slider_scale if positive_branch else -args.paired_slider_scale

        network.set_multiplier(0.0)
        if self.is_swapping_blocks:
            accelerator.unwrap_model(unet).prepare_block_swap_before_forward()
        with torch.no_grad(), accelerator.autocast():
            base_pred = unet(
                base_input,
                timesteps,
                prompt_embeds,
                padding_mask=padding_mask,
                source_attention_mask=attn_mask,
                t5_input_ids=t5_input_ids,
                t5_attn_mask=t5_attn_mask,
            )

        if args.gradient_checkpointing:
            model_input.requires_grad_(True)
            for condition in (prompt_embeds,):
                if condition is not None and condition.dtype.is_floating_point:
                    condition.requires_grad_(True)

        network.set_multiplier(signed_scale)
        self._paired_slider_multiplier_active = is_train
        if self.is_swapping_blocks:
            accelerator.unwrap_model(unet).prepare_block_swap_before_forward()
        with torch.set_grad_enabled(is_train), accelerator.autocast():
            model_pred = unet(
                model_input,
                timesteps,
                prompt_embeds,
                padding_mask=padding_mask,
                source_attention_mask=attn_mask,
                t5_input_ids=t5_input_ids,
                t5_attn_mask=t5_attn_mask,
            )

        if not is_train:
            network.set_multiplier(1.0)

        weighting = anima_train_utils.compute_loss_weighting_for_anima(
            weighting_scheme=args.weighting_scheme, sigmas=sigmas
        )
        return model_pred, base_pred.detach(), timesteps, weighting

    def process_batch(
        self, batch, text_encoders, unet, network, vae, noise_scheduler,
        vae_dtype, weight_dtype, accelerator, args,
        text_encoding_strategy, tokenize_strategy,
        is_train=True, train_text_encoder=True, train_unet=True,
    ) -> torch.Tensor:
        """Override base process_batch for 5D video latents (B, C, T, H, W).

        Base class assumes 4D (B, C, H, W) for loss.mean([1,2,3]) and weighting broadcast.
        """
        import typing
        from library.custom_train_functions import apply_masked_loss

        with torch.no_grad():
            if "latents" in batch and batch["latents"] is not None:
                latents = typing.cast(torch.FloatTensor, batch["latents"].to(accelerator.device))
            else:
                if args.vae_batch_size is None or len(batch["images"]) <= args.vae_batch_size:
                    latents = self.encode_images_to_latents(args, vae, batch["images"].to(accelerator.device, dtype=vae_dtype))
                else:
                    chunks = [
                        batch["images"][i : i + args.vae_batch_size] for i in range(0, len(batch["images"]), args.vae_batch_size)
                    ]
                    list_latents = []
                    for chunk in chunks:
                        with torch.no_grad():
                            chunk = self.encode_images_to_latents(args, vae, chunk.to(accelerator.device, dtype=vae_dtype))
                            list_latents.append(chunk)
                    latents = torch.cat(list_latents, dim=0)

                if torch.any(torch.isnan(latents)):
                    accelerator.print("NaN found in latents, replacing with zeros")
                    latents = typing.cast(torch.FloatTensor, torch.nan_to_num(latents, 0, out=latents))

            latents = self.shift_scale_latents(args, latents)

            reference_latents = None
            if getattr(args, "paired_difference_mode", False):
                conditioning_images = batch.get("conditioning_images")
                if conditioning_images is None:
                    raise ValueError(
                        "Paired Image Difference Mode expected conditioning_images. "
                        "Set conditioning_data_dir for every subset."
                    )
                if args.vae_batch_size is None or len(conditioning_images) <= args.vae_batch_size:
                    reference_latents = self.encode_images_to_latents(
                        args, vae, conditioning_images.to(accelerator.device, dtype=vae_dtype)
                    )
                else:
                    reference_latents = torch.cat(
                        [
                            self.encode_images_to_latents(
                                args,
                                vae,
                                conditioning_images[i : i + args.vae_batch_size].to(
                                    accelerator.device, dtype=vae_dtype
                                ),
                            )
                            for i in range(0, len(conditioning_images), args.vae_batch_size)
                        ],
                        dim=0,
                    )
                reference_latents = self.shift_scale_latents(args, reference_latents)
                if reference_latents.shape != latents.shape:
                    raise ValueError(
                        f"Paired latent shapes do not match: target={tuple(latents.shape)}, "
                        f"reference={tuple(reference_latents.shape)}"
                    )
                if torch.any(torch.isnan(reference_latents)):
                    accelerator.print("NaN found in paired reference latents, replacing with zeros")
                    reference_latents = torch.nan_to_num(reference_latents, 0)

        # Text encoder conditions
        text_encoder_conds = []
        text_encoder_outputs_list = batch.get("text_encoder_outputs_list", None)
        if text_encoder_outputs_list is not None:
            text_encoder_conds = text_encoding_strategy.drop_cached_text_encoder_outputs(
                *text_encoder_outputs_list
            )

        if len(text_encoder_conds) == 0 or text_encoder_conds[0] is None or train_text_encoder:
            with torch.set_grad_enabled(is_train and train_text_encoder), accelerator.autocast():
                input_ids = [ids.to(accelerator.device) for ids in batch["input_ids_list"]]
                encoded_text_encoder_conds = text_encoding_strategy.encode_tokens(
                    tokenize_strategy,
                    self.get_models_for_text_encoding(args, accelerator, text_encoders),
                    input_ids,
                )
                if args.full_fp16:
                    encoded_text_encoder_conds = [c.to(weight_dtype) for c in encoded_text_encoder_conds]

            if len(text_encoder_conds) == 0:
                text_encoder_conds = encoded_text_encoder_conds
            else:
                for i in range(len(encoded_text_encoder_conds)):
                    if encoded_text_encoder_conds[i] is not None:
                        text_encoder_conds[i] = encoded_text_encoder_conds[i]

        if getattr(args, "paired_difference_mode", False):
            noise_pred, target, timesteps, weighting = (
                self.get_paired_difference_predictions(
                    args,
                    accelerator,
                    latents,
                    reference_latents,
                    text_encoder_conds,
                    unet,
                    network,
                    weight_dtype,
                    is_train=is_train,
                )
            )
            huber_c = train_util.get_huber_threshold_if_needed(args, timesteps, noise_scheduler)
            loss = train_util.conditional_loss(
                noise_pred.float(), target.float(), args.loss_type, "none", huber_c
            )

            # Optionally focus both directions on regions that differ between
            # the aligned pair, while preserving some background supervision.
            if args.paired_difference_mask:
                change = (latents.float() - reference_latents.float()).abs().mean(dim=1, keepdim=True)
                spatial_mean = change.mean(
                    dim=tuple(range(2, change.ndim)), keepdim=True
                ).clamp_min(1e-6)
                soft_mask = change / (change + args.paired_mask_threshold * spatial_mean)
                mask_weight = args.paired_background_weight + (
                    1.0 - args.paired_background_weight
                ) * soft_mask
                if args.paired_mask_normalize:
                    # Preserve localized edit strength when the masked loss is
                    # later averaged over the complete latent canvas.
                    mask_mean = mask_weight.mean(
                        dim=tuple(range(1, mask_weight.ndim)), keepdim=True
                    ).clamp_min(1e-6)
                    mask_weight = mask_weight / mask_mean
                loss = loss * mask_weight
        else:
            noise_pred, target, timesteps, weighting = self.get_noise_pred_and_target(
                args, accelerator, noise_scheduler, latents, batch,
                text_encoder_conds, unet, network, weight_dtype, train_unet, is_train=is_train,
            )

            huber_c = train_util.get_huber_threshold_if_needed(args, timesteps, noise_scheduler)
            loss = train_util.conditional_loss(noise_pred.float(), target.float(), args.loss_type, "none", huber_c)

        if args.masked_loss or ("alpha_masks" in batch and batch["alpha_masks"] is not None):
            # WanVAE produces 5D latents [B,C,T,H,W] even for images (T=1).
            # Squeeze temporal dim so apply_masked_loss sees 4D [B,C,H,W].
            squeezed = loss.dim() == 5 and loss.shape[2] == 1
            if squeezed:
                loss = loss.squeeze(2)
            loss = apply_masked_loss(loss, batch)
            if squeezed:
                loss = loss.unsqueeze(2)

        # Reduce all non-batch dims: (B, C, T, H, W) -> (B,) for 5D, (B, C, H, W) -> (B,) for 4D
        reduce_dims = list(range(1, loss.ndim))
        loss = loss.mean(reduce_dims)

        # Apply weighting after reducing to (B,)
        if weighting is not None:
            loss = loss * weighting.reshape(weighting.shape[0], -1).mean(dim=1)

        loss_weights = batch["loss_weights"]
        loss = loss * loss_weights

        loss = self.post_process_loss(loss, args, timesteps, noise_scheduler)
        return loss.mean()

    def post_process_loss(self, loss, args, timesteps, noise_scheduler):
        return loss

    def get_sai_model_spec(self, args):
        return train_util.get_sai_model_spec(None, args, False, True, False, is_stable_diffusion_ckpt=True)

    def update_metadata(self, metadata, args):
        metadata["ss_weighting_scheme"] = args.weighting_scheme
        metadata["ss_discrete_flow_shift"] = args.discrete_flow_shift
        metadata["ss_timestep_sample_method"] = getattr(args, 'timestep_sample_method', 'logit_normal')
        metadata["ss_sigmoid_scale"] = getattr(args, 'sigmoid_scale', 1.0)
        metadata["ss_paired_difference_mode"] = bool(getattr(args, "paired_difference_mode", False))
        if getattr(args, "paired_difference_mode", False):
            metadata["ss_paired_objective"] = "addift_prediction_matching"
            metadata["ss_paired_slider_scale"] = args.paired_slider_scale
            metadata["ss_paired_min_timestep"] = args.paired_min_timestep
            metadata["ss_paired_max_timestep"] = args.paired_max_timestep
            metadata["ss_paired_difference_mask"] = args.paired_difference_mask
            metadata["ss_paired_mask_normalize"] = args.paired_mask_normalize
            metadata["ss_paired_background_weight"] = args.paired_background_weight

    def is_text_encoder_not_needed_for_training(self, args):
        return args.cache_text_encoder_outputs and not self.is_train_text_encoder(args)

    def prepare_unet_with_accelerator(
        self, args: argparse.Namespace, accelerator: Accelerator, unet: torch.nn.Module
    ) -> torch.nn.Module:
        # The base NetworkTrainer only calls enable_gradient_checkpointing(cpu_offload=True/False),
        # so we re-apply with unsloth_offload if needed (after base has already enabled it).
        if self._use_unsloth_offload_checkpointing and args.gradient_checkpointing:
            unet.enable_gradient_checkpointing(unsloth_offload=True)

        if not self.is_swapping_blocks:
            return super().prepare_unet_with_accelerator(args, accelerator, unet)

        dit = unet
        dit = accelerator.prepare(dit, device_placement=[not self.is_swapping_blocks])
        accelerator.unwrap_model(dit).move_to_device_except_swap_blocks(accelerator.device)
        accelerator.unwrap_model(dit).prepare_block_swap_before_forward()

        return dit

    def on_step_start(self, args, accelerator, network, text_encoders, unet, batch, weight_dtype, is_train=True):
        pass

    def on_after_backward(self, args, accelerator, network, text_encoders, unet, batch, weight_dtype):
        if self._paired_slider_multiplier_active:
            network.set_multiplier(1.0)
            self._paired_slider_multiplier_active = False

    def on_validation_step_end(self, args, accelerator, network, text_encoders, unet, batch, weight_dtype):
        if self.is_swapping_blocks:
            accelerator.unwrap_model(unet).prepare_block_swap_before_forward()


def setup_parser() -> argparse.ArgumentParser:
    parser = train_network.setup_parser()
    train_util.add_dit_training_arguments(parser)
    anima_train_utils.add_anima_training_arguments(parser)
    parser.add_argument(
        "--unsloth_offload_checkpointing",
        action="store_true",
        help="offload activations to CPU RAM using async non-blocking transfers (faster than --cpu_offload_checkpointing). "
        "Cannot be used with --cpu_offload_checkpointing or --blocks_to_swap.",
    )
    parser.add_argument(
        "--paired_difference_mode",
        action="store_true",
        help="train a LoRA from aligned target/reference pairs. image_dir contains targets and "
        "conditioning_data_dir contains clean references with matching filenames.",
    )
    parser.add_argument(
        "--paired_slider_scale",
        type=float,
        default=0.25,
        help="signed LoRA multiplier used during ADDifT-style paired training (default: 0.25)",
    )
    parser.add_argument(
        "--paired_min_timestep",
        type=int,
        default=500,
        help="minimum ADDifT timestep on a 0-1000 scale (default: 500)",
    )
    parser.add_argument(
        "--paired_max_timestep",
        type=int,
        default=1000,
        help="maximum ADDifT timestep on a 0-1000 scale (default: 1000)",
    )
    parser.add_argument(
        "--paired_direct_loss_weight",
        type=float,
        default=None,
        help=argparse.SUPPRESS,  # compatibility with configs created by the first experimental implementation
    )
    parser.add_argument(
        "--paired_difference_mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="softly emphasize latent regions that differ between paired images (default: enabled)",
    )
    parser.add_argument(
        "--paired_mask_normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="normalize each difference mask by effective area so small edits are not diluted (default: enabled)",
    )
    parser.add_argument(
        "--paired_mask_threshold",
        type=float,
        default=1.0,
        help="relative threshold used by the paired soft difference mask (default: 1.0)",
    )
    parser.add_argument(
        "--paired_background_weight",
        type=float,
        default=0.1,
        help="minimum loss weight outside changed regions, from 0 to 1 (default: 0.1)",
    )
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    trainer = AnimaNetworkTrainer()
    trainer.train(args)
