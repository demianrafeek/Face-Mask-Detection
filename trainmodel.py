
from detectionmodel import (FaceMaskDataset, Training)
import os
import torch 



epochs = 1
batch_Size = 2
data_path = 'G:\Arete\projects\\face_mask_detection\\test_data'
image_datasets = {x: FaceMaskDataset(root=os.path.join(data_path, x)) for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size= batch_Size,shuffle=True, collate_fn=image_datasets[x].collate_fn) for x in ['train', 'val']}
# dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Training()
results = model.train(dataloaders=dataloaders, device=device, epochs=epochs)

# plot train and val losses
model.plot_curve(results, epochs)

# save the trained model to dir
target_dir = 'G:\Arete\projects\\face_mask_detection\\test_data'
model.save_model(target_dir)

