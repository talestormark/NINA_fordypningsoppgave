"""
Example showing how to integrate WandbLogger into train.py.

Key changes needed:
1. Import WandbLogger
2. Add --wandb and --wandb-entity arguments
3. Initialize logger after creating output directory
4. Log metrics after each epoch
5. Finish logger at the end
"""

# 1. Add import at top of train.py
from logger import WandbLogger, create_run_name, create_tags

# 2. Add arguments in main() function (around line 519)
def main():
    parser = argparse.ArgumentParser(description='Train baseline land-take detection models')

    # ... existing arguments ...

    # Add these wandb arguments
    parser.add_argument('--wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='landtake-detection',
                        help='Weights & Biases project name')
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help='Weights & Biases team/user name (optional)')


# 3. Initialize logger in train() function (after creating output directory, around line 305)
def train(args):
    # ... existing code to create output directory and save config ...

    # Initialize wandb logger
    run_name = create_run_name(args.model_name, args.encoder_name, args.seed)
    tags = create_tags(
        model_name=args.model_name,
        encoder_name=args.encoder_name,
        loss=args.loss,
    )

    logger = WandbLogger(
        project=args.wandb_project,
        name=run_name,
        config=config,
        entity=args.wandb_entity,
        enabled=args.wandb,
        tags=tags,
        notes=f"Training {args.model_name} with {args.encoder_name} encoder (seed={args.seed})",
    )

    # ... existing code to create model ...

    # Optionally watch model gradients (after model creation, around line 347)
    logger.watch_model(model, log="gradients", log_freq=100)


# 4. Log metrics after each epoch (in training loop, around line 431)
def train(args):
    # ... training loop ...

    for epoch in range(start_epoch, args.epochs + 1):
        # Train
        train_metrics = train_one_epoch(...)

        # Validate
        val_metrics = validate(...)

        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Log to wandb
        logger.log_epoch(epoch, train_metrics, val_metrics, current_lr)

        # ... existing logging code ...


# 5. Finish logger at end of train() function (around line 476)
def train(args):
    # ... training code ...

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best validation IoU: {best_val_iou:.4f}")
    print(f"Checkpoints saved to: {output_dir}")
    print()

    # Finish wandb run
    logger.finish()
