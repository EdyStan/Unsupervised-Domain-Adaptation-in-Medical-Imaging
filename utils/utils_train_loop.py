import os
import time
import numpy as np
import torch

def train_da(source_train_loader, source_val_loader,
             target_train_loader, target_val_loader,
             model, optimizer, scheduler, max_epochs, root_dir,
             start_epoch=1, resume_training=False):

    best_val_loss = float('inf')

    print('resume_training:', resume_training)
    if resume_training:
        checkpoint = torch.load(os.path.join(root_dir, "checkpoint.pth"), map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        print(f"Resuming training from epoch {start_epoch} with best_val_loss = {best_val_loss:.4f}")

    device = next(model.parameters()).device  # assume model is already on the right device

    for epoch in range(start_epoch, max_epochs + 1):
        global_step = 0
        len_train_dataloader = min(len(source_train_loader), len(target_train_loader))
        last_percent = -1

        train_target_det_loss = train_det_loss = train_domain_loss = train_total_loss = 0.0

        print(f"\nEpoch {epoch}")
        print("Train:", end="", flush=True)

        model.train()
        epoch_train_start = time.time()

        for step, (source_batch, target_batch) in enumerate(zip(source_train_loader, target_train_loader)):
            optimizer.zero_grad()

            # Source batch
            source_images, source_targets = source_batch
            source_images = [img.to(device) for img in source_images]
            source_targets = [{k: v.to(device) for k, v in t.items()} for t in source_targets]

            # Target batch
            target_images, target_targets = target_batch
            target_images = [img.to(device) for img in target_images]
            target_targets = [{k: v.to(device) for k, v in t.items()} for t in target_targets]

            # GRL alpha
            p = float(global_step + epoch * len_train_dataloader) / (max_epochs * len_train_dataloader)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # Forward
            det_loss, dc_loss_source = model(source_images, source_targets, domain='source', alpha=alpha)
            tgt_det_loss, dc_loss_target = model(target_images, target_targets, domain='target', alpha=alpha)

            loss_dc = dc_loss_source + dc_loss_target
            total_loss = det_loss + loss_dc

            # Backward + Optimizer step
            total_loss.backward()
            optimizer.step()

            # Accumulate
            train_det_loss += det_loss.item()
            train_domain_loss += loss_dc.item()
            train_target_det_loss += tgt_det_loss.item()
            train_total_loss += total_loss.item()
            global_step += 1

            # Progress bar
            percent = int(100 * (step + 1) / len_train_dataloader)
            if percent != last_percent:
                print(f"{percent}%", end=' ', flush=True)
                last_percent = percent

        # Normalize
        num_train_steps = step + 1
        train_det_loss /= num_train_steps
        train_domain_loss /= num_train_steps
        train_target_det_loss /= num_train_steps
        train_total_loss /= num_train_steps

        epoch_train_end = time.time()
        elapsed = epoch_train_end - epoch_train_start
        print(f"\nEpoch {epoch} training time: {int(elapsed//60)}m {int(elapsed%60)}s")

        # Validation
        print("Val:", end="", flush=True)
        # model.eval()
        val_det_loss = val_domain_loss = val_target_det_loss = val_total_loss = 0.0
        len_val_dataloader = min(len(source_val_loader), len(target_val_loader))
        last_percent = -1
        epoch_val_start = time.time()

        with torch.no_grad():
            for step, (source_batch, target_batch) in enumerate(zip(source_val_loader, target_val_loader)):
                # Source batch
                source_images, source_targets = source_batch
                source_images = [img.to(device) for img in source_images]
                source_targets = [{k: v.to(device) for k, v in t.items()} for t in source_targets]

                # Target batch
                target_images, target_targets = target_batch
                target_images = [img.to(device) for img in target_images]
                target_targets = [{k: v.to(device) for k, v in t.items()} for t in target_targets]

                # Use the same last alpha or recompute if desired
                det_loss, dc_loss_source = model(source_images, source_targets, domain='source', alpha=alpha)
                tgt_det_loss, dc_loss_target = model(target_images, target_targets, domain='target', alpha=alpha)

                loss_dc = dc_loss_source + dc_loss_target
                total_loss = det_loss + loss_dc

                val_det_loss += det_loss.item()
                val_domain_loss += loss_dc.item()
                val_target_det_loss += tgt_det_loss.item()
                val_total_loss += total_loss.item()

                percent = int(100 * (step + 1) / len_val_dataloader)
                if percent != last_percent:
                    print(f"{percent}%", end=' ', flush=True)
                    last_percent = percent

        # Normalize validation
        num_val_steps = step + 1
        val_det_loss /= num_val_steps
        val_domain_loss /= num_val_steps
        val_target_det_loss /= num_val_steps
        val_total_loss /= num_val_steps

        epoch_val_end = time.time()
        elapsed = epoch_val_end - epoch_val_start
        print(f"\nEpoch {epoch} validation time: {int(elapsed//60)}m {int(elapsed%60)}s")

        # Logging
        print(f"\n[Epoch {epoch}]")
        print(f"Train Loss: {train_total_loss:.4f} | Detection: {train_det_loss:.4f}, Domain: {train_domain_loss:.4f}, Target Det: {train_target_det_loss:.4f}")
        print(f" Val  Loss: {val_total_loss:.4f} | Detection: {val_det_loss:.4f}, Domain: {val_domain_loss:.4f}, Target Det: {val_target_det_loss:.4f}")

        # Save best
        if val_total_loss < best_val_loss:
            print("Saving best model…")
            torch.save(model.state_dict(), os.path.join(root_dir, "best_uda_model.pth"))
            best_val_loss = val_total_loss

        # Checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss
        }, os.path.join(root_dir, "checkpoint.pth"))

        if scheduler:
            scheduler.step()

    return