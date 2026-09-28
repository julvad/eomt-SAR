import json
import os
import shutil
import torch
import torch.nn as nn
import torch.utils.data as DataLoader
import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import MulticlassF1Score, MulticlassJaccardIndex

from my_code_cleanrepo.utils.utils_plot import plot_tensorboard_batch_images
from my_code_cleanrepo.utils.losses import ComboLoss, WeightedComboLoss, MultiClassFocalLoss, FocalLoss
from typing import Literal


DICE_method = 'macro' # micro or macro ##NOTE: micro gives Dice>0.99??
CLASS_weights = [1.0, 3.0]

def save_and_manage_checkpoints(
    monitor_metric, monitor_value, 
    val_loss, iou_val, dice_val, 
    epoch, model, optimizer, 
    log_loss_fnc, log_transform, log_pretrained_weights, log_dir, log_dataset, 
    log_batch_size, log_frozen_backbone,
    best_models, max_models=3
):
    
    model_path = os.path.join(
        # log_dir, f"best_model_epoch_{epoch + 1}_{val_loss:.4f}"
        log_dir, f'best_model_epoch{epoch+1}_{monitor_metric}{monitor_value:.3f}.pth'
    )
    if log_loss_fnc == 'wce':
        log_loss_fnc = log_loss_fnc + f'_{CLASS_weights}'

    # TODO see if this saves
    torch.save(
        {
            "epoch": epoch+1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            'loss fnc': log_loss_fnc,
            'dataset': log_dataset,
            'batch size': log_batch_size,
            'transform': log_transform,
            'pretrained weights': log_pretrained_weights,
            'log_frozen_backbone': log_frozen_backbone,
            "val_loss": val_loss,
            'val_iou': iou_val,
            'val_dice': dice_val
        },
        model_path,
    )
        
    # Add new best model to the list
    best_models.append((monitor_value, epoch, model_path))
    
    # Sort and keep only the best max_models
    if monitor_metric=='loss':
        best_models.sort(key=lambda x: x[0], reverse=False)
    else:
        best_models.sort(key=lambda x: x[0], reverse=True) # reverse = True : from high to low

    while len(best_models) > max_models:
        worst = best_models.pop()  # Remove the last model in reverse sorted list (lowest IoU)
        try:
            if os.path.isfile(worst[2]):
                os.remove(worst[2])
            elif os.path.isdir(worst[2]):
                shutil.rmtree(worst[2])
            print(f"Removed old checkpoint: {worst[2]}")
        except Exception as e:
            print(f"Could not remove file: {worst[2]} ({e})")
    return best_models


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 25,
    lr: float = 1e-2,
    freeze_backbone: bool = False,
    loss_fnc:Literal['ce','wce','combo','wcombo','focal']='ce',
    num_classes: int = 2,
    weight_decay: float = 4e-2,
    monitor_metric:Literal['dice','iou','loss']='loss',
    log_dir: str = "runs",
    log_dataset:str = '512_20',
    log_batchsize:int=8,
    log_transform:str='sar_transform',
    log_pretrained_weights:str='imagenet',
    type_scheduler: Literal['reduce_lr','cosine_annealing'] = "reduce_lr",
    label_smoothing: bool = True,
    device: str = "cuda:0",
    n_tensorboard_plot:int=0,
    diff_lr_encoder_decoder_factor:bool=False,
) -> None:
    """
    Here,

    args:
    - data_mode (str) = 'pancro_duplication',
    - freeze_backbone: bool = False,
    num_classes (int): 4 or 6.
    - type_similarity (str): the type of similarity matrix we want to use. If 'transition', it means
        that it encodes the similarity in terms of how hard it is to transition from one class to the other
        and 'impossibility' encodes how two blobs can be different from one another ?
    """

    # assert num_classes == 2
    assert type_scheduler in ["reduce_lr", "cosine_annealing"]

    os.makedirs(log_dir, exist_ok=True)

# # set trainable params ### NOTE: for SMP.
    if freeze_backbone:
        for param in model.encoder.parameters():
            param.requires_grad = False
    else:
        for param in model.encoder.parameters():
            param.requires_grad = True

## LOSS function
    # basic ce
    if loss_fnc == 'ce':
        if label_smoothing:
            criterion= nn.CrossEntropyLoss(label_smoothing=0.1) ## TODO: Compare ignore_index=0?
        else:
            criterion = nn.CrossEntropyLoss()
    # weighted ce
    elif loss_fnc == 'wce': 
        # class_weights = compute_class_weights(train_loader, num_classes=num_classes)
        class_weights = torch.tensor(CLASS_weights)
        print("Class weights:", class_weights)
        if label_smoothing:
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.1) 
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    # ComboLoss (not weighted)
    elif loss_fnc == 'combo': 
        criterion = ComboLoss(alpha=0.5, smooth=1.0)
    #Weighted combo loss
    elif loss_fnc == 'wcombo': 
        # class_weights = compute_class_weights(train_loader, num_classes=num_classes)
        class_weights = torch.tensor(CLASS_weights)
        print("Class weights:", class_weights)
        criterion = WeightedComboLoss(alpha=0.5, smooth=1.0,class_weights=class_weights.to(device))
    elif loss_fnc == 'focal':
        alpha = torch.tensor(CLASS_weights) 
        # criterion = MultiClassFocalLoss(alpha=alpha.to(device))
        criterion = FocalLoss(alpha=alpha.to(device))
    else: 
        raise ValueError('loss_fnc')


    writer = SummaryWriter(log_dir)

    if diff_lr_encoder_decoder_factor:
        optimizer = torch.optim.AdamW(
            [
                {"params": model.decoder.parameters(), "lr": lr*diff_lr_encoder_decoder_factor},
            ],
            weight_decay=0.05, #TODO: see if this is better
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if type_scheduler == "reduce_lr":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=4, factor=0.6
        )
    elif type_scheduler == "cosine_annealing":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs
        )
    else:
        raise ValueError(f"Scheduler type {type_scheduler} not yet supported")

    # Print the number of trainable parameters:
    print(
        f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    # On initialise la meilleure validation loss
    best_val_loss = float("inf")
    best_val_metric = 0
    best_models = []


    iou_train = MulticlassJaccardIndex(num_classes=num_classes, average=DICE_method).to(device)
    dice_train = MulticlassF1Score(num_classes=num_classes, average=DICE_method).to(device)
    iou_val = MulticlassJaccardIndex(num_classes=num_classes, average=DICE_method).to(device)
    dice_val = MulticlassF1Score(num_classes=num_classes, average=DICE_method).to(device)

    global_step = 0
    for epoch in range(num_epochs):
        # Training loop
        model.train()
        iou_train.reset()
        dice_train.reset()
        train_loss = 0

        for images, masks in tqdm.tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            images, masks = images.to(device), masks.to(device)
            # Forward pass
            # logits = model(images) # original code
            logits = predict_eomt(eomt_model=model, batch_tensor=images, pred_mask_size=model.rsz_size)

            loss = criterion(logits, masks)

            # Backward pass
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            predicted_mask = logits.argmax(dim=1)
            iou_train.update(predicted_mask, masks)
            dice_train.update(predicted_mask, masks)
            
            # Log training loss every 25 batches (or 200)
            if global_step % 200 == 0:
                writer.add_scalar("Loss/train_step", loss.item(), global_step)
                writer.add_scalar(
                    "IOU/train_step", iou_train.compute().item(), global_step
                )
                writer.add_scalar(
                    "Dice/train_step", dice_train.compute().item(), global_step
                )

            global_step += 1

        avg_train_loss = train_loss / len(train_loader)

            
        if n_tensorboard_plot>0:
            plot_tensorboard_batch_images(
                writer, images, masks, predicted_mask, epoch, name_tensorboard="train",n_images=n_tensorboard_plot, num_classes=num_classes
            )

        # Validation loop
        model.eval()
        val_loss = 0
        iou_val.reset()
        dice_val.reset()
        first_val_batch = None
        with torch.no_grad():
            for images, masks in tqdm.tqdm(val_loader, desc=f"Val loop: epoch {epoch + 1}"):
                images, masks = images.to(device), masks.to(device)

                # logits = model(images) #original pytorch code
                logits = predict_eomt(eomt_model=model, batch_tensor=images, pred_mask_size=model.rsz_size)

                loss = criterion(logits, masks)
                val_loss += loss.item()

                predicted_mask = logits.argmax(dim=1)

                if first_val_batch is None:
                    first_val_batch = (
                        images.cpu(),
                        masks.cpu(),
                        predicted_mask.cpu(),
                    )
                            
                iou_val.update(predicted_mask, masks)
                dice_val.update(predicted_mask, masks)

        avg_val_loss = val_loss / len(val_loader)
        if type_scheduler == 'cosine_annealing':
            scheduler.step()
        else:  
            scheduler.step(avg_val_loss)
        
        images, masks, predicted_mask = first_val_batch
        if n_tensorboard_plot>0:
            plot_tensorboard_batch_images( # plot last batch
                writer, images, masks, predicted_mask, epoch, name_tensorboard='val',n_images=n_tensorboard_plot, num_classes=num_classes
            )

        iou_val_epoch = iou_val.compute().item()
        dice_val_epoch = dice_val.compute().item()

        # Log epoch metrics
        writer.add_scalar("IOU/val_epoch", iou_val_epoch, epoch)
        writer.add_scalar("Dice/val_epoch", dice_val_epoch, epoch)
        writer.add_scalar("Loss/train_epoch", avg_train_loss, epoch)
        writer.add_scalar("Loss/val_epoch", avg_val_loss, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        print(f"Epoch {epoch + 1}/{num_epochs}:")
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print('IoU_val:', iou_val_epoch)
        print('Dice_val:', dice_val_epoch)
        print(f'tensorboard --logdir={os.path.abspath(log_dir)}')
        # if avg_val_loss < best_val_loss or len(best_models) < 4: ## galeio code = Monitor loss
        
        if monitor_metric == 'iou':
            monitor_value = iou_val_epoch
        elif monitor_metric == 'dice':
            monitor_value = dice_val_epoch
        elif monitor_metric == 'loss':
            monitor_value = avg_val_loss
        else:
            raise ValueError(monitor_metric)

        if (monitor_metric in ['iou','dice'] and monitor_value > best_val_metric) or (monitor_metric=='loss' and monitor_value < best_val_metric) or len(best_models) < 3: 
            best_models = save_and_manage_checkpoints(
                monitor_metric=monitor_metric,
                monitor_value=monitor_value,
                val_loss=avg_val_loss,
                iou_val=iou_val_epoch,
                dice_val=dice_val_epoch,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                log_loss_fnc=loss_fnc,
                log_transform=log_transform,
                log_pretrained_weights=log_pretrained_weights,
                log_dir=log_dir,
                log_dataset=log_dataset,
                log_batch_size=log_batchsize,
                log_frozen_backbone=freeze_backbone,
                best_models=best_models,
                max_models=3,
            )
            best_val_metric = min([x[0] for x in best_models]) if monitor_metric=='loss' else max([x[0] for x in best_models])
            # best_val_metric = max([x[0] for x in best_models]) # update best model iou

        torch.cuda.empty_cache()

    writer.close()

    # Save a simple json with the val loss
    url_json_file = os.path.join(log_dir, "model_metadata.json")
    with open(url_json_file, "w") as f:
        json.dump(
            {
                "val_loss": best_val_loss,
                'loss_fnc': loss_fnc,
                "freeze_backbone": freeze_backbone,
                "num_epochs": num_epochs,
                "lr": lr,
                "weight_decay": weight_decay,
                "type_scheduler": type_scheduler,
            },
            f,
        )
    
    
    del logits, predicted_mask, images, masks
    torch.cuda.empty_cache()


def predict_eomt(eomt_model, batch_tensor, pred_mask_size):
    masks_queries_logits, class_queries_logits = eomt_model(batch_tensor) 

    # Last decoder layer
    mask_logits = masks_queries_logits[-1]   # [B,Q,56,56] for 224 224 input
    class_logits = class_queries_logits[-1] # [B,Q,3]

    # Remove "no-object" class
    class_probs = class_logits.softmax(dim=-1)[..., :-1]

    # Query masks + query classes -> semantic logits
    semantic_logits = torch.einsum(
        "bqhw,bqc->bchw",
        mask_logits,
        class_probs,
    )

    # Upsample to original image size
    logits = nn.functional.interpolate(
        semantic_logits,
        size=pred_mask_size,
        mode="bilinear",
        align_corners=False,
    ) 

    return logits