
from detectionmodel import FaceMaskDataset,Testing
import os
import torch 

batch_Size = 8
data_path = 'G:\Arete\projects\\face_mask_detection\\faceMask'
image_datasets = {'test': FaceMaskDataset(root=os.path.join(data_path, 'test'))}
dataloaders = {'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size= batch_Size,shuffle=True, collate_fn=image_datasets['test'].collate_fn)}
# dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_path = 'G:\Arete\projects\\face_mask_detection\\faceMask\\model_epoch_25'
model = Testing(model_path, device)

images, boxes, labels = next(iter(dataloaders['test']))
images = list(img.to(device) for img in images)

predictions = model.predict(images)
print(predictions)

for i in range(len(predictions)):
  model.display_images(images= model.display_boundary(images[i], predictions[i]['boxes'], predictions[i]['labels'], predictions[i]['scores']))

# print(type(images[0]))
# model.display_images(Example1=model.display_boundary(images[0], boxes[0], labels[0],.89),)
#         # Example2=model.display_boundary(images[1], boxes[1], labels[1]))

