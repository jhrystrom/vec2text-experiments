from vec2text.run_args import ModelArguments, DataArguments, TrainingArguments
from vec2text.experiments import experiment_from_args
from transformers.trainer_pt_utils import AcceleratorConfig
from vec2text.utils.init_codebook import initialize_model_codebook_from_dataset

from transformers.trainer_utils import get_last_checkpoint

model_args = ModelArguments(
    model_name_or_path="t5-base",
    embedder_model_name="gtr_base",
    embedder_model_api=None,
    embedder_gaussian_noise_level=0.0,
    embedder_torch_dtype="float32",
    embedding_transform_strategy="repeat",
    encoder_dropout_disabled=False,
    decoder_dropout_disabled=False,
    model_type=None,
    config_overrides=None,
    config_name=None,
    tokenizer_name=None,
    cache_dir=None,
    model_revision="main",
    max_seq_length=32,
    torch_dtype=None,
    num_repeat_tokens=16,
    embedding_zero_except_topk=None,
    embedder_no_grad=True,
    use_lora=False,
    embedder_fake_with_zeros=False,
    use_frozen_embeddings_as_input=True,
    corrector_ignore_hypothesis_embedding=False,
    embeddings_from_layer_n=None,
    freeze_strategy="none",
)
data_args = DataArguments(dataset_name="msmarco", max_eval_samples=500, use_less_data=-1)
training_args = TrainingArguments(
    output_dir="./saves/gtr-1",
    overwrite_output_dir=False,
    do_train=False,
    do_eval=False,
    do_predict=False,
    eval_strategy="steps",
    prediction_loss_only=False,
    per_device_train_batch_size=512,
    per_device_eval_batch_size=512,
    per_gpu_train_batch_size=None,
    per_gpu_eval_batch_size=None,
    gradient_accumulation_steps=1,
    eval_accumulation_steps=None,
    eval_delay=0,
    torch_empty_cache_steps=None,
    learning_rate=0.001,
    weight_decay=0.0,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-06,
    max_grad_norm=1.0,
    num_train_epochs=100.0,
    max_steps=-1,
    lr_scheduler_type="constant_with_warmup",
    lr_scheduler_kwargs={},
    warmup_ratio=0.0,
    warmup_steps=156,
    log_level="passive",
    log_level_replica="warning",
    log_on_each_node=True,
    logging_dir="./saves/gtr-1/runs/Mar11_14-24-22_virgil",
    logging_strategy="steps",
    logging_first_step=False,
    logging_steps=6,
    logging_nan_inf_filter=True,
    save_strategy="steps",
    save_steps=312,
    save_total_limit=2,
    save_safetensors=True,
    save_on_each_node=False,
    save_only_model=False,
    restore_callback_states_from_checkpoint=False,
    no_cuda=False,
    use_cpu=False,
    use_mps_device=False,
    seed=42,
    data_seed=None,
    jit_mode_eval=False,
    use_ipex=False,
    bf16=True,
    fp16=False,
    fp16_opt_level="O1",
    half_precision_backend="auto",
    bf16_full_eval=False,
    fp16_full_eval=False,
    tf32=None,
    local_rank=0,
    ddp_backend=None,
    tpu_num_cores=None,
    tpu_metrics_debug=False,
    debug=[],
    dataloader_drop_last=False,
    eval_steps=312,
    dataloader_num_workers=4,
    dataloader_prefetch_factor=None,
    past_index=-1,
    run_name="./saves/gtr-1",
    disable_tqdm=True,
    remove_unused_columns=False,
    label_names=None,
    load_best_model_at_end=True,
    metric_for_best_model=None,
    greater_is_better=False,
    ignore_data_skip=False,
    fsdp=[],
    fsdp_min_num_params=0,
    fsdp_config={
        "min_num_params": 0,
        "xla": False,
        "xla_fsdp_v2": False,
        "xla_fsdp_grad_ckpt": False,
    },
    fsdp_transformer_layer_cls_to_wrap=None,
    accelerator_config=AcceleratorConfig(
        split_batches=False,
        dispatch_batches=None,
        even_batches=True,
        use_seedable_sampler=True,
        non_blocking=False,
        gradient_accumulation_kwargs=None,
        use_configured_state=False,
    ),
    deepspeed=None,
    label_smoothing_factor=0.0,
    optim="adamw_torch",
    optim_args=None,
    adafactor=False,
    group_by_length=True,
    length_column_name="length",
    report_to=["wandb"],
    ddp_find_unused_parameters=False,
    ddp_bucket_cap_mb=None,
    ddp_broadcast_buffers=None,
    dataloader_pin_memory=True,
    dataloader_persistent_workers=False,
    skip_memory_metrics=True,
    use_legacy_prediction_loop=False,
    push_to_hub=False,
    resume_from_checkpoint=None,
    hub_model_id=None,
    hub_strategy="every_save",
    hub_token=None,
    hub_private_repo=None,
    hub_always_push=False,
    gradient_checkpointing=False,
    gradient_checkpointing_kwargs=None,
    include_inputs_for_metrics=True,
    include_for_metrics=["inputs"],
    eval_do_concat_batches=True,
    fp16_backend="auto",
    evaluation_strategy="steps",
    push_to_hub_model_id=None,
    push_to_hub_organization=None,
    push_to_hub_token=None,
    mp_parameters="",
    auto_find_batch_size=False,
    full_determinism=False,
    torchdynamo=None,
    ray_scope="last",
    ddp_timeout=1800,
    torch_compile=True,
    torch_compile_backend="inductor",
    torch_compile_mode=None,
    dispatch_batches=None,
    split_batches=None,
    include_tokens_per_second=False,
    include_num_input_tokens_seen=False,
    neftune_noise_alpha=None,
    optim_target_modules=None,
    batch_eval_metrics=False,
    eval_on_start=False,
    use_liger_kernel=False,
    eval_use_gather_object=False,
    average_tokens_across_devices=False,
    corrector_model_alias=None,
    corrector_model_from_pretrained=None,
    cheat_on_train_hypotheses=False,
    steps_per_epoch=500000,
    use_wandb=True,
    experiment="inversion",
    exp_name="",
    exp_group_name="oct-gtr",
    mock_embedder=False,
)
experiment = experiment_from_args(
    model_args=model_args, data_args=data_args, training_args=training_args
)


last_checkpoint = get_last_checkpoint("vec2text/saves/gtr-1")


model = experiment.load_model()
trainer = experiment.load_trainer()
trainer._load_from_checkpoint(last_checkpoint)
trained_codebook = model.vector_quantizer.codebook

initialize_model_codebook_from_dataset(model, train_dataset=trainer.train_dataset)

original_codebook = model.vector_quantizer.codebook
trained_codebook == original_codebook
