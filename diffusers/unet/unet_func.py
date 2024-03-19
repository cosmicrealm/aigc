import diffusers
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from typing import Optional, Tuple, Union
import torch
from diffusers.models.attention_processor import Attention

if __name__ == "__main__":
    print("This is the main function of unet_func.py")
    args = {
        "sample_size": None,
        "in_channels": 4,
        "out_channels": 4,
        "center_input_sample": False,
        "flip_sin_to_cos": True,
        "freq_shift": 0,
        "down_block_types": (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        "mid_block_type": "UNetMidBlock2DCrossAttn",
        "up_block_types": (
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
        ),
        "only_cross_attention": False,
        "block_out_channels": (320, 640, 1280, 1280),
        "layers_per_block": 2,
        "downsample_padding": 1,
        "mid_block_scale_factor": 1,
        "dropout": 0.0,
        "act_fn": "silu",
        "norm_num_groups": 32,
        "norm_eps": 1e-5,
        "cross_attention_dim": 1280,
        "transformer_layers_per_block": 1,
        "reverse_transformer_layers_per_block": None,
        "encoder_hid_dim": None,
        "encoder_hid_dim_type": None,
        "attention_head_dim": 8,
        "num_attention_heads": None,
        "dual_cross_attention": False,
        "use_linear_projection": False,
        "class_embed_type": None,
        "addition_embed_type": None,
        "addition_time_embed_dim": None,
        "num_class_embeds": None,
        "upcast_attention": False,
        "resnet_time_scale_shift": "default",
        "resnet_skip_time_act": False,
        "resnet_out_scale_factor": 1.0,
        "time_embedding_type": "positional",
        "time_embedding_dim": None,
        "time_embedding_act_fn": None,
        "timestep_post_act": None,
        "time_cond_proj_dim": None,
        "conv_in_kernel": 3,
        "conv_out_kernel": 3,
        "projection_class_embeddings_input_dim": None,
        "attention_type": "default",
        "class_embeddings_concat": False,
        "mid_block_only_cross_attention": None,
        "cross_attention_norm": None,
        "addition_embed_type_num_heads": 64,
    }

    save_ckpt = False
    forward_input = False
    get_method_name = False
    atten_info_parse = True

    unet = UNet2DConditionModel(**args)

    if save_ckpt:
        ckpt_fname = "unet_2d_condition_ckpt.pth"
        torch.save(unet.state_dict(), ckpt_fname)

    if forward_input:
        device = "cuda" if torch.cuda.is_available() else "mps"
        unet = unet.to(device)
        sample = torch.randn(1, 4, 64, 64)
        sample = sample.to(device)
        timestep = 10
        cross_attention_dim = 1280
        sequence_length = 10
        encoder_hidden_states = torch.randn(1, sequence_length, cross_attention_dim).to(
            device
        )
        output = unet(sample, timestep, encoder_hidden_states)
        print(output.sample.shape)

    if get_method_name:
        """
        __call__
        __class__
        __delattr__
        __dir__
        __eq__
        __format__
        __ge__
        __getattr__
        __getattribute__
        __gt__
        __hash__
        __init__
        __init_subclass__
        __le__
        __lt__
        __ne__
        __new__
        __reduce__
        __reduce_ex__
        __repr__
        __setattr__
        __setstate__
        __sizeof__
        __str__
        __subclasshook__
        _apply
        _call_impl
        _check_config
        _convert_deprecated_attention_blocks
        _convert_ip_adapter_attn_to_diffusers
        _convert_ip_adapter_image_proj_to_diffusers
        _dict_from_json_file
        _fuse_lora_apply
        _get_backward_hooks
        _get_backward_pre_hooks
        _get_init_keys
        _get_name
        _load_from_state_dict
        _load_ip_adapter_weights
        _load_pretrained_model
        _maybe_warn_non_full_backward_hook
        _named_members
        _register_load_state_dict_pre_hook
        _register_state_dict_hook
        _replicate_for_data_parallel
        _save_to_state_dict
        _set_add_embedding
        _set_class_embedding
        _set_encoder_hid_proj
        _set_gradient_checkpointing
        _set_pos_net_if_use_gligen
        _set_time_proj
        _slow_forward
        _temp_convert_self_to_deprecated_attention_blocks
        _undo_temp_convert_self_to_deprecated_attention_blocks
        _unfuse_lora_apply
        _upload_folder
        active_adapters
        add_adapter
        add_module
        apply
        bfloat16
        buffers
        children
        convert_state_dict_legacy_attn_format
        cpu
        cuda
        delete_adapters
        disable_adapters
        disable_freeu
        disable_gradient_checkpointing
        disable_lora
        disable_xformers_memory_efficient_attention
        double
        enable_adapters
        enable_freeu
        enable_gradient_checkpointing
        enable_lora
        enable_xformers_memory_efficient_attention
        eval
        extra_repr
        extract_init_dict
        float
        forward
        from_config
        from_pretrained
        fuse_lora
        fuse_qkv_projections
        get_aug_embed
        get_buffer
        get_class_embed
        get_config_dict
        get_extra_state
        get_parameter
        get_submodule
        get_time_embed
        half
        ipu
        load_attn_procs
        load_config
        load_state_dict
        modules
        named_buffers
        named_children
        named_modules
        named_parameters
        num_parameters
        parameters
        process_encoder_hidden_states
        push_to_hub
        register_backward_hook
        register_buffer
        register_forward_hook
        register_forward_pre_hook
        register_full_backward_hook
        register_full_backward_pre_hook
        register_load_state_dict_post_hook
        register_module
        register_parameter
        register_state_dict_pre_hook
        register_to_config
        requires_grad_
        save_attn_procs
        save_config
        save_pretrained
        set_adapter
        set_adapters
        set_attention_slice
        set_attn_processor
        set_default_attn_processor
        set_extra_state
        set_use_memory_efficient_attention_xformers
        share_memory
        state_dict
        to
        to_empty
        to_json_file
        to_json_string
        train
        type
        unfuse_lora
        unfuse_qkv_projections
        unload_lora
        xpu
        zero_grad
        """
        methods = [
            method_name
            for method_name in dir(UNet2DConditionModel)
            if callable(getattr(UNet2DConditionModel, method_name))
        ]
        for method in methods:
            print(method)

    if atten_info_parse:
        # get attention block info
        for k, attention_model in unet.attn_processors.items():
            cross_attention_dim = (
                None
                if k.endswith("attn1.processor")
                else unet.config.cross_attention_dim
            )
            # 构建attention_model 的输入
            attn = Attention(query_dim=10, cross_attention_dim=19)
            hidden_states = torch.randn(1, 12, 10)
            encoder_hidden_states = torch.randn(1, 12, 19)
            output = attention_model(attn, hidden_states, encoder_hidden_states)
            print(k, output.shape, cross_attention_dim)
            if cross_attention_dim:
                layer_name = k.split(".processor")[0]
                weights = {
                    "to_k_ip.weight": unet.state_dict()[layer_name + ".to_k.weight"],
                    "to_v_ip.weight": unet.state_dict()[layer_name + ".to_v.weight"],
                }
                print(weights["to_k_ip.weight"].shape)
                print(weights["to_v_ip.weight"].shape)
            # down_blocks.0.attentions.0.transformer_blocks.0.attn1.processor torch.Size([1, 12, 10])
        """
                down_blocks.0.attentions.0.transformer_blocks.0.attn1.processor <diffusers.models.attention_processor.AttnProcessor2_0 object at 0x7f8c301155b0>
                down_blocks.0.attentions.0.transformer_blocks.0.attn2.processor <diffusers.models.attention_processor.AttnProcessor2_0 object at 0x7f8c30115760>
                down_blocks.0.attentions.1.transformer_blocks.0.attn1.processor <diffusers.models.attention_processor.AttnProcessor2_0 object at 0x7f8c30115cd0>
                down_blocks.0.attentions.1.transformer_blocks.0.attn2.processor <diffusers.models.attention_processor.AttnProcessor2_0 object at 0x7f8c30115e80>
                down_blocks.1.attentions.0.transformer_blocks.0.attn1.processor <diffusers.models.attention_processor.AttnProcessor2_0 object at 0x7f8c31671580>
                down_blocks.1.attentions.0.transformer_blocks.0.attn2.processor <diffusers.models.attention_processor.AttnProcessor2_0 object at 0x7f8c31671730>
                down_blocks.1.attentions.1.transformer_blocks.0.attn1.processor <diffusers.models.attention_processor.AttnProcessor2_0 object at 0x7f8c31671ca0>
                down_blocks.1.attentions.1.transformer_blocks.0.attn2.processor <diffusers.models.attention_processor.AttnProcessor2_0 object at 0x7f8c31671e50>
                down_blocks.2.attentions.0.transformer_blocks.0.attn1.processor <diffusers.models.attention_processor.AttnProcessor2_0 object at 0x7f8c49812550>
                down_blocks.2.attentions.0.transformer_blocks.0.attn2.processor <diffusers.models.attention_processor.AttnProcessor2_0 object at 0x7f8c49812700>
                down_blocks.2.attentions.1.transformer_blocks.0.attn1.processor <diffusers.models.attention_processor.AttnProcessor2_0 object at 0x7f8c49812cd0>
                down_blocks.2.attentions.1.transformer_blocks.0.attn2.processor <diffusers.models.attention_processor.AttnProcessor2_0 object at 0x7f8c49812e80>
                up_blocks.1.attentions.0.transformer_blocks.0.attn1.processor <diffusers.models.attention_processor.AttnProcessor2_0 object at 0x7f8c27c8b850>
                up_blocks.1.attentions.0.transformer_blocks.0.attn2.processor <diffusers.models.attention_processor.AttnProcessor2_0 object at 0x7f8c27c8ba00>
                up_blocks.1.attentions.1.transformer_blocks.0.attn1.processor <diffusers.models.attention_processor.AttnProcessor2_0 object at 0x7f8c27c90070>
                up_blocks.1.attentions.1.transformer_blocks.0.attn2.processor <diffusers.models.attention_processor.AttnProcessor2_0 object at 0x7f8c27c90220>
                up_blocks.1.attentions.2.transformer_blocks.0.attn1.processor <diffusers.models.attention_processor.AttnProcessor2_0 object at 0x7f8c27c90850>
                up_blocks.1.attentions.2.transformer_blocks.0.attn2.processor <diffusers.models.attention_processor.AttnProcessor2_0 object at 0x7f8c27c90a00>
                up_blocks.2.attentions.0.transformer_blocks.0.attn1.processor <diffusers.models.attention_processor.AttnProcessor2_0 object at 0x7f8c17d62190>
                up_blocks.2.attentions.0.transformer_blocks.0.attn2.processor <diffusers.models.attention_processor.AttnProcessor2_0 object at 0x7f8c17d62340>
                up_blocks.2.attentions.1.transformer_blocks.0.attn1.processor <diffusers.models.attention_processor.AttnProcessor2_0 object at 0x7f8c17d62970>
                up_blocks.2.attentions.1.transformer_blocks.0.attn2.processor <diffusers.models.attention_processor.AttnProcessor2_0 object at 0x7f8c17d62b20>
                up_blocks.2.attentions.2.transformer_blocks.0.attn1.processor <diffusers.models.attention_processor.AttnProcessor2_0 object at 0x7f8c17f54040>
                up_blocks.2.attentions.2.transformer_blocks.0.attn2.processor <diffusers.models.attention_processor.AttnProcessor2_0 object at 0x7f8c17f54220>
                up_blocks.3.attentions.0.transformer_blocks.0.attn1.processor <diffusers.models.attention_processor.AttnProcessor2_0 object at 0x7f8c17f54970>
                up_blocks.3.attentions.0.transformer_blocks.0.attn2.processor <diffusers.models.attention_processor.AttnProcessor2_0 object at 0x7f8c17f54b20>
                up_blocks.3.attentions.1.transformer_blocks.0.attn1.processor <diffusers.models.attention_processor.AttnProcessor2_0 object at 0x7f8c5395b190>
                up_blocks.3.attentions.1.transformer_blocks.0.attn2.processor <diffusers.models.attention_processor.AttnProcessor2_0 object at 0x7f8c5395b340>
                up_blocks.3.attentions.2.transformer_blocks.0.attn1.processor <diffusers.models.attention_processor.AttnProcessor2_0 object at 0x7f8c5395b970>
                up_blocks.3.attentions.2.transformer_blocks.0.attn2.processor <diffusers.models.attention_processor.AttnProcessor2_0 object at 0x7f8c5395bb20>
                mid_block.attentions.0.transformer_blocks.0.attn1.processor <diffusers.models.attention_processor.AttnProcessor2_0 object at 0x7f8c42d0b8b0>
                mid_block.attentions.0.transformer_blocks.0.attn2.processor <diffusers.models.attention_processor.AttnProcessor2_0 object at 0x7f8c42d0ba60>
        """
