import numpy as np
import pandas as pd
import torch
import torchvision
import time
import copy
import cv2
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageFont
import os
import albumentations as A
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch.nn.functional as F
from pathlib import Path
import time
# import utils
from tqdm import trange
import torch.optim as optim
from torch.optim import lr_scheduler
from torch import nn



class FaceMaskDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms

        self.imgs = list(sorted(os.listdir(os.path.join(root, 'images'))))
        self.anns = list(sorted(os.listdir(os.path.join(root, 'annotations'))))
        self.img_dir = os.path.join(root, 'images')
        self.ann_dir  = os.path.join(root, 'annotations')


    def __len__(self):
        return len(self.imgs)
        
    def get_annotations_boxes_from_xml(self,dir):
        tree = ET.parse(dir)
        root = tree.getroot()

        annotations, labels = [], []

        for neighbor in root.iter('bndbox'):
            xmin = int(neighbor.find('xmin').text)
            ymin = int(neighbor.find('ymin').text)
            xmax = int(neighbor.find('xmax').text)
            ymax = int(neighbor.find('ymax').text)

            annotations.append([xmin, ymin, xmax, ymax])

        for neighbor in root.iter('object'):
            label = neighbor.find('name').text
            if label == 'with_mask':
                labels.append(1)
            elif label == 'without_mask':
                labels.append(2)
            else:
                labels.append(3)

        return annotations, labels

    def __getitem__(self, idx):
        curr_img_dir = os.path.join(self.img_dir, self.imgs[idx])
        curr_ann_dir = os.path.join(self.ann_dir, self.anns[idx])

        image = Image.open(curr_img_dir, mode='r').convert('RGB')
        boxes, labels = self.get_annotations_boxes_from_xml(curr_ann_dir)

        boxes = torch.FloatTensor(boxes)
        labels = torch.LongTensor(labels)

        if self.transforms:
            transformed = self.transforms(image=image, bboxes=boxes, category_ids=labels)

        tenn = transforms.ToTensor()
        image = tenn(image)

        return image, boxes, labels

    def collate_fn(self, batch):
        return tuple(zip(*batch))
    
 

class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 dataloaders: torch.utils.data.DataLoader,
                 epochs: int,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler,
                 device: torch.device):

        self.model = model
        self.train_dataloader = dataloaders['train']
        self.val_dataloader = dataloaders['val']
        self.epochs = epochs
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.epoch = 0
        
        # Create empty results dictionary
        self.results = {"train_loss": [],
                        "val_loss": []}

    def train_model(self):
        """
        Train the Model.
        """
        start_time = time.time()

        progressbar = trange(self.epochs, desc="Progress")
        for _ in progressbar:
            # Epoch counter
            self.epoch += 1
            #progressbar.set_description(f"Epoch {self.epoch}")

            # Training block
            self.train_epoch()
            self.val_epoch()
            print(f'\nEpoch {self.epoch}: Train loss: {self.results["train_loss"][-1]}, Val loss: {self.results["val_loss"][-1]}')

            # Save checkpoints every epoch

        time_elapsed = time.time() - start_time
        print('\n')
        print('-' * 20)
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

        # plot training curve
        # plot_curve(results=self.results, epochs=self.epochs)

        return self.results

    def train_epoch(self):
        """
        Training Mode
        """
        self.model.train() # training mode
        running_losses = []

        for images, boxes, labels in self.train_dataloader:
            # Send to device (GPU or CPU)
            images = list(img.to(self.device) for img in images)
            boxes = [b.to(self.device) for b in boxes]
            labels = [l.to(self.device) for l in labels]
            targets = []

            for i in range(len(images)):
                d = {}
                d['boxes'] = boxes[i]
                d['labels'] = labels[i]
                targets.append(d)

            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward - track history if only in train
            loss_dict = self.model(images, targets)
            # Calculate the loss
            loss = sum(loss for loss in loss_dict.values())
            loss_value = loss.item() 
            running_losses.append(loss_value)

            # Backward pass
            loss.backward()
            # Update the parameters
            self.optimizer.step()

        self.scheduler.step()
        self.results["train_loss"].append(np.mean(running_losses))

    def val_epoch(self):
        """
        Validation Mode
        """
        # self.model.eval() # Validation mode
        running_losses = []

        for images, boxes, labels in self.val_dataloader:
            # Send to device (GPU or CPU)
            images = list(img.to(self.device) for img in images)
            boxes = [b.to(self.device) for b in boxes]
            labels = [l.to(self.device) for l in labels]
            targets = []

            for i in range(len(images)):
                d = {}
                d['boxes'] = boxes[i]
                d['labels'] = labels[i]
                targets.append(d)

            with torch.no_grad():
                loss_dict = self.model(images, targets)
                # Calculate the loss
                loss = sum(loss for loss in loss_dict.values())
                loss_value = loss.item()
                running_losses.append(loss_value)

        self.results["val_loss"].append(np.mean(running_losses))



class FaceMaskeDetection:

    def __init__(self):

        # Model Loading 
        self.weights = torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        self.model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=self.weights)
        self.in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(self.in_features, 4)


    def display_boundary(self, image, boxes, labels, score = None):

        label_to_name = {1: 'Masked', 2: 'No Mask', 3: 'Masked_incorrectly'}
        label_to_color = {1: 'palegreen', 2: 'red', 3: 'blue'}

        transform = torchvision.transforms.ToPILImage()
        image = transform(image)
        # image = torchvision.transforms.ToPILImage()(image)
        boxes = boxes.tolist()
        labels = labels.tolist()

        img_bbox = ImageDraw.Draw(image)
        new_font = ImageFont.truetype("arial.ttf", 15)#(os.path.join(cv2.__path__[0],'qt','fonts','DejaVuSansCondensed-Bold.ttf'), 10)

        for idx in range(len(boxes)):
            img_bbox.rectangle(boxes[idx], outline=label_to_color[labels[idx]], width=2)
            if score == None:
                img_bbox.text((boxes[idx][0], boxes[idx][1]-15), label_to_name[labels[idx]],
                            font=new_font, align ="left", fill=label_to_color[labels[idx]])
            else:
                img_bbox.text((boxes[idx][0], boxes[idx][1]-15), label_to_name[labels[idx]]+' '+ f"{score[idx].item():.2%}",
                            font=new_font, align ="left", fill=label_to_color[labels[idx]])

        return image

    # helper function for image visualization
    def display_images(self, **images):
        """
        Plot images in one rown
        """
        
        num_images = len(images)
        plt.figure(figsize=(15,15))
        for idx, (name, image) in enumerate(images.items()):
            plt.subplot(1, num_images, idx + 1)
            plt.xticks([])
            plt.yticks([])
            # get title from the parameter names
            plt.title(name.replace('_',' ').title(), fontsize=15)
            plt.imshow(image)
            plt.show()




class Training(FaceMaskeDetection):
    def __init__(self):
        super().__init__()


    def train(self, dataloaders, device, epochs=25):

        ## Model inItialization
        self.model = self.model.to(device)
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer_RCNN = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
        exp_lr_scheduler_RCNN = lr_scheduler.StepLR(optimizer_RCNN, step_size=7, gamma=0.1)

        # Trainer
        trainer = Trainer(model=self.model,
                        dataloaders=dataloaders,
                        epochs=epochs,
                        optimizer=optimizer_RCNN,
                        scheduler=exp_lr_scheduler_RCNN,
                        device=device)
                        
        self.results = trainer.train_model()

        return self.results


    
    # Save the model to the target dir
    def save_model(self, target_dir: str):
        """
        Saves a PyTorch model to a target directory.
        """ 
        # Create target directory
        target_dir_path = Path(target_dir)
        target_dir_path.mkdir(parents=True, exist_ok=True)

        # Create model save path
        check_point_name = f"model"
        model_save_path = target_dir_path / check_point_name

        # Save the model state_dict()
        #print(f"[INFO] Saving model to: {model_save_path}")
        torch.save(obj=self.model.state_dict(), f=model_save_path)

    # Plot the training curve
    def plot_curve(self, results: dict, epochs: int):
        #train_ious, val_ious = np.array(results["train_iou"]), np.array(results["val_iou"])
        train_losses = np.array(results["train_loss"])
        val_losses = np.array(results["val_loss"])

        plt.plot(np.arange(epochs, step=1), train_losses, label='Train loss')
        plt.plot(np.arange(epochs, step=1), val_losses, label='Val loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper right')
        plt.show()

    def bbox_iou(self, box1, box2):
        """
        Returns the IoU of two bounding boxes
        """
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

        # get the corrdinates of the intersection rectangle
        inter_rect_x1 = torch.max(b1_x1, b2_x1)
        inter_rect_y1 = torch.max(b1_y1, b2_y1)
        inter_rect_x2 = torch.min(b1_x2, b2_x2)
        inter_rect_y2 = torch.min(b1_y2, b2_y2)
        # Intersection area
        inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
            inter_rect_y2 - inter_rect_y1 + 1, min=0
        )
        # Union Area
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

        iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

        return iou       
 

class Testing(FaceMaskeDetection):
    def __init__(self, model_path, device):
        super().__init__()
        self.model_path = model_path
        self.device = device
        # pass        
        ## Unet effb4 batch size 16 lr 0.001 argumentation
        ## Model inItialization
    # def load(self, model_path, device):
        model_state = torch.load(self.model_path,map_location=torch.device(self.device))
        self.model.load_state_dict(model_state)
        self.model = self.model.to(self.device)
        self.model.eval()

    def remove_low_risk_box(self, predictions, threshold: float):

        for img in range(len(predictions)):
            for idx in range(predictions[img]['labels'].shape[0]):
                if predictions[img]['scores'][idx] < threshold:
                    predictions[img]['boxes'] = predictions[img]['boxes'][0:idx]
                    predictions[img]['labels'] = predictions[img]['labels'][0:idx]
                    predictions[img]['scores'] = predictions[img]['scores'][0:idx]
                    break

        return predictions

    def nms_pytorch(self, P : torch.tensor, labels: torch.tensor, scores: torch.tensor, thresh_iou : float):
        """
        Apply non-maximum suppression to avoid detecting too many
        overlapping bounding boxes for a given object.
        Args:
            boxes: (tensor) The location preds for the image
                along with the class predscores, Shape: [num_boxes,5].
            thresh_iou: (float) The overlap thresh for suppressing unnecessary boxes.
        Returns:
            A list of filtered boxes, Shape: [ , 5]
        """

        # we extract coordinates for every
        # prediction box present in P
        x1 = P[:, 0]
        y1 = P[:, 1]
        x2 = P[:, 2]
        y2 = P[:, 3]

        # we extract the confidence scores as well
        #scores = P[:, 4]

        # calculate area of every block in P
        areas = (x2 - x1) * (y2 - y1)

        # sort the prediction boxes in P
        # according to their confidence scores
        order = scores.argsort()

        # initialise an empty list for
        # filtered prediction boxes
        pred_dict = {'boxes':[], 'labels':[], 'scores':[]}

        while len(order) > 0:

            # extract the index of the
            # prediction with highest score
            # we call this prediction S
            idx = order[-1]

            # push S in filtered predictions list
            if len(pred_dict['boxes']) == 0:
                pred_dict['boxes'].append(torch.unsqueeze(P[idx], dim=0))
                pred_dict['labels'].append(torch.unsqueeze(labels[idx], dim=0))
                pred_dict['scores'].append(torch.unsqueeze(scores[idx], dim=0))
            else:
                pred_dict['boxes'][0] = torch.cat((pred_dict['boxes'][0], torch.unsqueeze(P[idx], dim=0)), dim=0)
                pred_dict['labels'][0] = torch.cat((pred_dict['labels'][0], torch.unsqueeze(labels[idx], dim=0)), dim=0)
                pred_dict['scores'][0] = torch.cat((pred_dict['scores'][0], torch.unsqueeze(scores[idx], dim=0)), dim=0)

            # remove S from P
            order = order[:-1]

            # sanity check
            if len(order) == 0:
                break

            # select coordinates of BBoxes according to
            # the indices in order
            xx1 = torch.index_select(x1,dim = 0, index = order)
            xx2 = torch.index_select(x2,dim = 0, index = order)
            yy1 = torch.index_select(y1,dim = 0, index = order)
            yy2 = torch.index_select(y2,dim = 0, index = order)

            # find the coordinates of the intersection boxes
            xx1 = torch.max(xx1, x1[idx])
            yy1 = torch.max(yy1, y1[idx])
            xx2 = torch.min(xx2, x2[idx])
            yy2 = torch.min(yy2, y2[idx])

            # find height and width of the intersection boxes
            w = xx2 - xx1
            h = yy2 - yy1

            # take max with 0.0 to avoid negative w and h
            # due to non-overlapping boxes
            w = torch.clamp(w, min=0.0)
            h = torch.clamp(h, min=0.0)

            # find the intersection area
            inter = w*h

            # find the areas of BBoxes according the indices in order
            rem_areas = torch.index_select(areas, dim = 0, index = order)

            # find the union of every prediction T in P
            # with the prediction S
            # Note that areas[idx] represents area of S
            union = (rem_areas - inter) + areas[idx]

            # find the IoU of every prediction in P with S
            IoU = inter / union
            # keep the boxes with IoU less than thresh_iou
            mask = IoU < thresh_iou
            order = order[mask]

        if len(pred_dict['labels'])>0:
            pred_dict['boxes'], pred_dict['labels'], pred_dict['scores'] = pred_dict['boxes'][0], pred_dict['labels'][0], pred_dict['scores'][0]
        return pred_dict

    def apply_nms(self, predictions, threshold):
        nms_list = []
        for img in range(len(predictions)):
            nms_list.append(self.nms_pytorch(predictions[img]['boxes'], predictions[img]['labels'], predictions[img]['scores'], threshold))
        return nms_list
    
    def predict(self,images):

        self.model.eval()
        predictions = self.model(images)
        predictions = self.remove_low_risk_box(predictions, threshold=0.5)
        predictions = self.apply_nms(predictions, 0.5)

        return predictions
