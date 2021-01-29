import os
from collections import defaultdict
from typing import Dict, Any
import logging
import copy

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from datasets import EastDataset, get_filenames_for_this_stage
from loss import TowerLoss, PODLoss
from model import East


def train(config: Dict[str, Any]):

    device = torch.device(config['device'])

    filenames_df = ... # here goes reading from dataiku

    train_df = filenames_df[filenames_df['split'] == "train"]
    valid_df = filenames_df[filenames_df['split'] == "valid"]

    for stage in range(config['total_num_stages']):

        train_image_filenames, train_boxes_filenames, train_memory_flags = get_filenames_for_this_stage(train_df,
                                                                                                        stage=stage)
        valid_image_filenames, valid_boxes_filenames, valid_memory_flags = get_filenames_for_this_stage(valid_df,
                                                                                                        stage=stage)

        dataset_train = EastDataset(img_paths=train_image_filenames,
                                    label_paths=train_boxes_filenames,
                                    memory_flags=train_memory_flags,
                                    size=config['target_img_size'])

        dataset_valid = EastDataset(img_paths=valid_image_filenames,
                                    label_paths=valid_boxes_filenames,
                                    memory_flags=valid_memory_flags,
                                    size=config['target_img_size'])

        train_loader = DataLoader(dataset_train,
                                  batch_size=config['batch_size'],
                                  shuffle=True,
                                  num_workers=config['num_workers'])

        valid_loader = DataLoader(dataset_valid,
                                  batch_size=config['batch_size'],
                                  shuffle=False,
                                  num_workers=config['num_workers'])
        if stage == 0:
            model = East()
            model.to(device)
        else:
            teacher_model = East()
            teacher_model.load_state_dict(copy.deepcopy(model.state_dict()))
            teacher_model.to(device)
        task_loss_criterion = TowerLoss()
        distillation_loss_criterion = PODLoss(
            height_coef=config['pod_height_coef'],
            width_coef=config['pod_width_coef'],
            flat_coef=config['pod_flat_coef']
        )

        optimizer = Adam(model.parameters(), lr=config['lr'])
        best_val_loss = float("inf")

        for epoch in range(config['num_epochs']):
            log_train = defaultdict(list)
            for batch_number, input_data in enumerate(train_loader):

                input_images = input_data['images'].to(model.device())
                input_scores = input_data['score_maps'].to(model.device())
                input_geometry = input_data['geo_maps'].to(model.device())
                input_masks = input_data['training_masks'].to(model.device())
                input_memory_flags = input_data['memory_flags'].to(model.device())

                featuremaps, predicted_scores, predicted_geometry = model(input_images)
                loss = task_loss_criterion(input_scores,
                                           predicted_scores,
                                           input_geometry,
                                           predicted_geometry,
                                           input_masks)

                log_train['east'].append(loss.item())
                if stage > 0:
                    with torch.no_grad():
                        teacher_featuremaps, teacher_scores, teacher_geometry = teacher_model(input_images)
                        width_loss, height_loss, flat_loss = distillation_loss_criterion(teacher_featuremaps, featuremaps,
                                                                        teacher_scores, predicted_scores, input_memory_flags)
                        log_train['width'].append(width_loss)
                        log_train['height'].append(height_loss)
                        log_train['flat'].append(flat_loss)
                        loss += config['distillation_loss_coef'] * (height_loss+width_loss+flat_loss)

                loss.backward()
                optimizer.step()

                if batch_number % config['log_every_n'] == 0:
                    message = f"Train Epoch {epoch}: " \
                              f"{' '.join([f'{key}:{torch.mean(torch.tensor(value))}' for key, value in log_train.items()])}"
                    logging.info(message)

            log_valid = defaultdict(list)
            for batch_index, input_data in enumerate(valid_loader):
                input_images = input_data['images'].to(model.device)
                input_scores = input_data['scores'].to(model.device)
                input_geometry = input_data['geometry'].to(model.device)
                input_masks = input_data['masks'].to(model.device)
                with torch.no_grad():
                    featuremaps, predicted_scores, predicted_geometry = model(input_images)
                    loss = task_loss_criterion(input_scores,
                                               predicted_scores,
                                               input_geometry,
                                               predicted_geometry,
                                               input_masks)
                    log_valid['east'].append(loss.item())
                    if stage != 0:
                        with torch.no_grad():
                            teacher_featuremaps, teacher_scores, teacher_geometry = teacher_model(input_images)
                            width_loss, height_loss, flat_loss = distillation_loss_criterion(teacher_featuremaps,
                                                                                             featuremaps,
                                                                                             teacher_scores,
                                                                                             predicted_scores)
                            log_valid['width'].append(width_loss)
                            log_valid['height'].append(height_loss)
                            log_valid['flat'].append(flat_loss)

                    mean_val_loss = torch.mean(torch.tensor(log_valid['east']))

            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                model_filename = f"east_stage_{stage}_epoch_{epoch}_loss_{best_val_loss.item:.4f}.pth"
                model_filepath = os.path.join(config['model_save_path'], model_filename)
                torch.save(model.state_dict(), model_filepath)

            message = f"Valid Epoch {epoch}: {' '.join([f'{key}:{torch.mean(torch.tensor(value))}' for key, value in log_valid.items()])}"
            logging.info(message)

if __name__ == "__main__":
    config = {
        'total_num_stages':...,
        'target_img_size': 512,
        'device': "cuda",
        'batch_size': 2,
        'num_workers': 8,
        'pod_height_coef': 1.,
        'pod_width_coef': 1.,
        'pod_flat_coef': 1.,
        'lr': 0.001,
        'num_epoch': 10,
        'distillation_loss_coef': 0.1,
    }