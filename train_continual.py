from collections import defaultdict
from typing import Dict, Any
import logging

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from data_utils import OcrDataset
from loss import TowerLoss, PODLoss
from model import East


def train(config: Dict[str, Any]):

    dataset_train = OcrDataset() # todo
    dataset_valid = OcrDataset() # todo
    train_loader = DataLoader(dataset_train,
                              batch_size=config['batch_size'],
                              shuffle=True,
                              num_workers=config['num_workers'])
    valid_loader = DataLoader(dataset_valid,
                              batch_size=config['batch_size'],
                              shuffle=False,
                              num_workers=config['num_workers'])

    model = East()
    if config['cuda_devices']:
        # TODO verify that model works both in dp, single-gpu and cpu setups
        model = nn.DataParallel(model, device_ids=config['cuda_devices'])
        model.cuda()
    teacher_model = East() if config['stage'] == 0 else None
    task_loss_criterion = TowerLoss()
    distillation_loss_criterion = PODLoss(
        height_coef=config['pod_height_coef'],
        width_coef=config['pod_width_coef'],
        flat_coef=config['pod_flat_coef']
    )

    optimizer = Adam(model.parameters(), lr=config['lr'])
    for epoch in range(config['num_epochs']):
        log_train = defaultdict(list)
        for batch_number, input_data in enumerate(train_loader):

            input_images = input_data['images'].to(model.device())
            input_scores = input_data['scores'].to(model.device())
            input_geometry = input_data['geometry'].to(model.device())
            input_masks = input_data['masks'].to(model.device())

            featuremaps, predicted_scores, predicted_geometry = model(input_images)
            loss = task_loss_criterion(input_scores,
                                       predicted_scores,
                                       input_geometry,
                                       predicted_geometry,
                                       input_masks)

            log_train['east'].append(loss.item())
            if config['stage'] != 0:
                with torch.no_grad():
                    teacher_featuremaps, teacher_scores, teacher_geometry = teacher_model(input_images)
                    width_loss, height_loss, flat_loss = distillation_loss_criterion(teacher_featuremaps, featuremaps,
                                                                    teacher_scores, predicted_scores)
                    log_train['width'].append(width_loss)
                    log_train['height'].append(height_loss)
                    log_train['flat'].append(flat_loss)
                    loss += config['distillation_loss_coef'] * (height_loss+width_loss+flat_loss)

            loss.backward()
            optimizer.step()

            if batch_number % config['log_every_n'] == 0:
                message = f"Train Epoch {epoch}: {' '.join([f'{key}:{torch.mean(torch.tensor(value))}' for key, value in log_train.items()])}"
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
                if config['stage'] != 0:
                    with torch.no_grad():
                        teacher_featuremaps, teacher_scores, teacher_geometry = teacher_model(input_images)
                        width_loss, height_loss, flat_loss = distillation_loss_criterion(teacher_featuremaps,
                                                                                         featuremaps,
                                                                                         teacher_scores,
                                                                                         predicted_scores)
                        log_valid['width'].append(width_loss)
                        log_valid['height'].append(height_loss)
                        log_valid['flat'].append(flat_loss)
        message = f"Valid Epoch {epoch}: {' '.join([f'{key}:{torch.mean(torch.tensor(value))}' for key, value in log_valid.items()])}"
        logging.log(message)
        # todo add saving code (based on dataiku.get_upload_stream)