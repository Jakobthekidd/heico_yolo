source main.env
# Sync YOLO project code to cluster
rsync -azvh \
    --exclude=__pycache__ \
    --exclude=.vscode \
    --exclude=outputs \
    --exclude=wandb_runs \
    --exclude=.idea \
    --exclude=.lr* \
    --exclude=.pytest_cache \
    --exclude=.git \
    --exclude=*.ckpt \
    --exclude=wandb \
    --exclude=unsloth_compile_cache \
    --exclude=outputs_model \
    --exclude=figures \
    --exclude=cache \
    --exclude=unsloth_compiled_cache \
    $LOCAL_CODE_PATH/YOLO/heico_code  \
    $SUBMISSION_EMAIL:.
