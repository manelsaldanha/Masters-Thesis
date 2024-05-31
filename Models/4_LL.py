import os
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
import clip
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from peft import LoraConfig, TaskType,get_peft_model
from transformers import AutoTokenizer,AutoProcessor
from HF_CLIP import CLIPTextModel, CLIPVisionModel, CLIPVisionModelWithProjection, CLIPTextModelWithProjection
import json
import torchvision
from torchvision.ops.boxes import _box_inter_union
from tqdm import tqdm

def get_IoU(pred, gt):

    x1_pred,y1_pred,x2_pred,y2_pred=pred.unbind(1)
    x1_gt,y1_gt,x2_gt,y2_gt=gt.unbind(1)

    total_elements=len(x1_pred)

    # determine the coordinates of the intersection rectangle
    x_left = torch.max(x1_pred, x1_gt)
    y_top = torch.max(y1_pred, y1_gt)
    x_right = torch.min(x2_pred, x2_gt)
    y_bottom = torch.min(y2_pred, y2_gt)

    not_overlapping_condition=(x_right < x_left) | (y_bottom < y_top)

    not_overlapping_boxes_index=torch.nonzero(not_overlapping_condition).squeeze()
    overlapping_boxes_index=torch.nonzero(~not_overlapping_condition).squeeze()

    x_right_filtered = torch.masked_select(x_right, ~not_overlapping_condition)
    x_left_filtered = torch.masked_select(x_left, ~not_overlapping_condition)
    y_bottom_filtered = torch.masked_select(y_bottom, ~not_overlapping_condition)
    y_top_filtered = torch.masked_select(y_top, ~not_overlapping_condition)

    intersection_area = (x_right_filtered - x_left_filtered) * (y_bottom_filtered - y_top_filtered)

    bb1_area = (torch.masked_select(x2_pred, ~not_overlapping_condition) - torch.masked_select(x1_pred, ~not_overlapping_condition)) * (torch.masked_select(y2_pred, ~not_overlapping_condition) - torch.masked_select(y1_pred, ~not_overlapping_condition))
    bb2_area = (torch.masked_select(x2_gt, ~not_overlapping_condition) - torch.masked_select(x1_gt, ~not_overlapping_condition)) * (torch.masked_select(y2_gt, ~not_overlapping_condition) - torch.masked_select(y1_gt, ~not_overlapping_condition))
    iou = intersection_area / (bb1_area + bb2_area - intersection_area)
    avg_iou=torch.sum(iou).item()/total_elements*100

    return avg_iou

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x[:,0,:].unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size, mode):
    img_w, img_h = size
    e = torch.stack([img_w, img_h, img_w, img_h], dim=1)
    e = e.to(device)
    if mode=='down':
        b = out_bbox / e
    if mode=='up':
        b = out_bbox * e
    return b

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print,best_score=None):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = best_score
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = val_loss
        
        self.trace_func(f'best score: {self.best_score}')

        if self.best_score is None:
            self.trace_func('self.best_score is None')
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'learning_rate': optimizer.param_groups[0]['lr']
            }, self.path)

        self.val_loss_min = val_loss

class Boxalignment(nn.Module):
    def __init__(self):
        super(Boxalignment, self).__init__()

    def forward(self, x):
        x1_pred,y1_pred,x2_pred,y2_pred=x.unbind(1)

        x1= torch.min(x1_pred,x2_pred)
        y1= torch.min(y1_pred,y2_pred)
        x2= torch.max(x1_pred,x2_pred)
        y2= torch.max(y1_pred,y2_pred)

        return torch.stack([x1, y1, x2, y2], dim=1)

class Model(nn.Module):
    def __init__(self,hidden_dim=768, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()

        target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc_in", "fc_out"] 
        peft_config = LoraConfig(inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1,target_modules=target_modules)

        #CLIPVision with LoRA
        self.vit=get_peft_model(CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14"),peft_config)
        
        self.v_patches_linear=nn.Linear(in_features=self.vit.model.visual_projection.in_features, out_features=self.vit.model.visual_projection.out_features, bias=False)
        with torch.no_grad():
            self.v_patches_linear.weight.copy_(self.vit.model.visual_projection.weight)

        #CLIPText with LoRA
        self.text_encoder=get_peft_model(CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14"),peft_config)

        self.t_tokens_linear=nn.Linear(in_features=self.text_encoder.model.text_projection.in_features, out_features=self.text_encoder.model.text_projection.out_features, bias=False)
        with torch.no_grad():
            self.t_tokens_linear.weight.copy_(self.text_encoder.model.text_projection.weight)
        
        #Transformer decoder from torch
        self.decoder=nn.Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers, batch_first=True).decoder

        #linear layer converts hidden layer dim to dim 4 (because of how a bounding box is define) 
        self.linear_bbox= nn.Linear(hidden_dim, 4)
        
        self.bbox_align=Boxalignment()
        #torch tensor created randomly 
        single_tensor = torch.empty(1,6,hidden_dim)
        nn.init.xavier_normal_(single_tensor)

        #creating queries as a parameter of the model using the single tensor variable
        self.queries = nn.Parameter(single_tensor, requires_grad=True)


    def forward(self, image: torch.Tensor, text: torch.Tensor):
        
        #output tokens of CLIPVision [64,50,512] -> [batch_size, nr_patches+cls, hidden_dim]
        i_cls=self.vit(image)
                
        i_patches=self.v_patches_linear(i_cls['last_hidden_state'])
        
        i=torch.cat((i_cls['image_embeds'], i_patches), dim=1)
        
        #output tokens of CLIPText [64,25,512] -> [batch_size, nr_tokens_in_the_biggest_sentence_in_the_batch, hidden_dim]
        #the token sentences in the batch are padded into the biggest token sentence in the batch
        t_cls=self.text_encoder(**text)

        t_tokens=self.t_tokens_linear(t_cls['last_hidden_state'])

        t=torch.cat((t_cls['text_embeds'], t_tokens), dim=1)

        #masking the padded tokens from CLIP text
        i_padding_mask=torch.full(size=(i.shape[0],i.shape[1]), fill_value=False, dtype=torch.bool).to(device)
        t_padding_mask=~text['attention_mask'].to(torch.bool)
        memory_padding_mask=torch.cat((i_padding_mask, t_padding_mask), dim=1)

        #concat output tokens of CLIPVision and CLIPText
        memory=torch.cat((i, t), dim=1)

        #get the size of the batch that comes from the CLIP
        memory_batch_size,_,_=memory.shape
        
        #send the queries multiplied by the memory_batch_size to the decoder input, send the CLIP output to the decoder cross-attention layer
        #send the respective mask that hides the padded text tokens from CLIP
        h=self.decoder(self.queries.repeat(memory_batch_size,1, 1), memory ,memory_key_padding_mask=memory_padding_mask)
        
        pred_unfiltered=self.linear_bbox(h).sigmoid()

        max_token=pred_unfiltered.shape[1]-1
        final_pred=self.bbox_align(pred_unfiltered[:,max_token,:])
        #the output of the decoder goes through is converted into a tensor of size 4 and a sigmoid function is applied in order to get
        #the normalized coordinates of the perdicted bounding box
        return {'pred_boxes': final_pred}

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, annotations_file, mode, transform=None, target_transform=None):


        json_file_path = os.path.join(annotations_file,mode+'.json')
        
        # Open the file and load the JSON data
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)
        
        self.annotations=data
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):

        img_path = os.path.join(self.img_dir, self.annotations[idx]['filename'])
        image = Image.open(img_path)
        im_size=(self.annotations[idx]['width'],self.annotations[idx]['height'])
    
        label = self.annotations[idx]['bbox']
        question = self.annotations[idx]['refer']
        if self.transform:
            image = self.transform(images=image,return_tensors="pt",size=(224,224))['pixel_values'].squeeze(0)
            #to_pil = ToPILImage()
            #pil_image = to_pil(image['pixel_values'].squeeze(0))
            #pil_image.save(r"C:\Users\manue\OneDrive\Ambiente de Trabalho\Tese\DeTR\CLIP_HF\output_image.jpg")
        if self.target_transform:
            label = self.target_transform(label)
        return image, label, question, im_size, img_path

class Loss(nn.Module):
    def __init__(self, reduction_type='mean',mse_weight=2,giou_weight=5):
        super().__init__()
        self.MSE = nn.MSELoss(reduction=reduction_type)
        self.alpha=mse_weight
        self.beta=giou_weight
        self.reduction_type=reduction_type

    def forward(self, predicted: torch.Tensor, gt: torch.Tensor):
        # Calculate the loss
        Mse=self.MSE(predicted, gt)
        GIoU=torchvision.ops.generalized_box_iou_loss(predicted,gt,reduction=self.reduction_type)

        loss=Mse*self.alpha+GIoU*self.beta

        return loss


def train_one_epoch():
    #Falta ver o input img esta [64,1,3,224,224] tem que ser [64,3,224,224], e a questao tem que ser codificada
    running_loss = 0.
    last_loss = 0.
    avg_iou=0.0

    batch_predictions=[]
    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(tqdm(train_dataloader)):
        # Every data instance is an input + label pair
        img, gt, question,im_size, img_path = data

        text = tokenizer(question, padding=True, return_tensors="pt").to(device)
        img=img.to(device)
        gt=torch.stack(gt, dim=-1).to(device)
        #im_size=torch.stack(im_size, dim=-1)
        # Zero your gradients for every batch!
        optimizer.zero_grad()
        #with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        # Make predictions for this batch
        outputs = model(img,text) 
        #scale up outputs['pred_boxes'][0] for image prediction drawing
        batch_predictions.append({"img_path":img_path[0],"question":question[0],"GT":gt[0].tolist(),"Pred":rescale_bboxes(outputs['pred_boxes'],im_size,'up')[0].tolist()})
        #scale down gt for loss calculation
        gt_normalized = rescale_bboxes(gt,im_size,'down').to(torch.float32)
        # Compute the loss and its gradients
        loss=criterion(outputs['pred_boxes'], gt_normalized)
        #loss=giou_loss(outputs['pred_boxes'][:,max_token,:],gt_normalized)
        #print('batch',i+1)
        #print('Loss',loss.item()) 
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        #print(loss.item())
        #print(i)
        #if i % 10 == 9:
        #last_loss = running_loss #/ 10 # loss per batch
        #print('  batch {} loss: {}'.format(i + 1, last_loss))
        #print(f'gt {gt}, question {question}, im_size {im_size}')
    
        avg_iou+=get_IoU(outputs['pred_boxes'], gt_normalized)
        #print('AIoU',avg_iou)
        #print('training AIoU: ',avg_iou/(i+1))

    return running_loss/(i+1), avg_iou/(i+1), batch_predictions

def val_one_epoch():
    v_img_batch_pred=[]
    running_vloss = 0.0
    avg_v_iou=0.0
    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(val_dataloader):
            vimg, vgt, vquestion,vim_size,vimg_path = vdata
            vtext = tokenizer(vquestion, padding=True, return_tensors="pt").to(device)
            vimg=vimg.to(device)
            vgt=torch.stack(vgt, dim=-1).to(device)

            # Make predictions for this batch
            voutputs = model(vimg,vtext)

            v_img_batch_pred.append({"img_path":vimg_path[0],"question":vquestion[0],"GT":vgt[0].tolist(),"Pred":rescale_bboxes(voutputs['pred_boxes'],vim_size,'up')[0].tolist()})

            vgt_normalized= rescale_bboxes(vgt,vim_size,'down')
            vloss=criterion(voutputs['pred_boxes'], vgt_normalized)
            running_vloss += vloss
            avg_v_iou += get_IoU(voutputs['pred_boxes'], vgt_normalized)
    avg_vloss = running_vloss / (i + 1)

    return avg_vloss, avg_v_iou/(i+1), v_img_batch_pred

def write_to_file(filename, text,header=None):
    try:
        # Try to open the file in read mode to check if it exists
        with open(filename, 'r') as file:
            existing_content = file.read()
        
        # If the file exists, open it in append mode to continue writing
        with open(filename, 'a') as file:
            # Check if the existing content is empty
            if existing_content.strip():
                # If not empty, add a new line before writing new text
                file.write('\n')
            # Write the new text
            file.write(text)
    except:
        # If the file doesn't exist, create a new one and write to it
        with open(filename, 'w') as file:
            if header:
                file.write(f'{header}' + "\n")
            file.write(text)

def write_to_json(filename, data):
    try:
        # Try to open the file in read mode to check if it exists
        with open(filename, 'r') as json_file:
            existing_data = json.load(json_file)
        
        # If the file exists, open it in write mode to update data
        with open(filename, 'w') as json_file:
            # Combine existing data with new data
            if existing_data:
                existing_data.append(data)
                json.dump(existing_data, json_file)
            else:
                json.dump([data], json_file)
    except:
        # If the file doesn't exist, create a new one and write to it
        with open(filename, 'w') as json_file:
            json.dump([data], json_file)


def draw_bbox(predictions):

    first_last=[predictions[0],predictions[-1]]

    # Define a color palette with 13 different colors (RGB format)

    save_folder=r'/cfs/home/u021542/model/CLIP_HF/final_results/new_multi_loss_draw'
    questions=[]
    images=[]
    GTs=[]
    first_pred=[]
    last_pred=[]

    for i,epoch in enumerate(first_last):
        for batch in epoch:
            if i==0:
                image = Image.open(batch[0])
                if len(np.array(image).shape)!=3:
                    image = image.convert("RGB")
            else:
                im_path=os.path.join(save_folder,batch[0].split('/')[-1])
                image = Image.open(im_path)
            draw = ImageDraw.Draw(image)
            question=batch[1]
            gt_x1,gt_y1,gt_x2,gt_y2=batch[2]
            p_x1,p_y1,p_x2,p_y2=batch[3]
            if i==0:
                draw.rectangle([gt_x1, gt_y1, gt_x2, gt_y2], outline=(0, 255, 0), width=2)
                draw.rectangle([p_x1, p_y1, p_x2, p_y2], outline=(0, 0, 225), width=2)
                
                questions.append(batch[1])
                GTs.append(batch[2])
                first_pred.append(batch[3])
                
            else:
                draw.rectangle([p_x1, p_y1, p_x2, p_y2], outline=(255, 0, 0), width=2)
                last_pred.append(batch[3])
            
            save_path=os.path.join(save_folder,batch[0].split('/')[-1])
            images.append(save_path)
            image.save(save_path)
    
    with open(os.path.join(save_folder,'predictions.txt'), 'w') as file:

        file.write('Image question GT first_prediction last_prediction\n')
        
        for a, b, c, d, e in zip(images,questions,GTs,first_pred,last_pred):
            file.write(f'{a} {b} {c} {d} {e}' + "\n")

    return

torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

batch_size=64

epoch_number = 0

EPOCHS = 150

tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")

img_folder=r'/tmp/Toloka_dataset/complete_DS/images'
ann_file = r'/tmp/Toloka_dataset/complete_DS/annotations_unc'
train_dataset = CustomImageDataset(img_folder,ann_file,'train',processor)
val_dataset = CustomImageDataset(img_folder,ann_file,'val',processor)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

model=Model().to(device)

#model.load_state_dict(torch.load('results/M_6input_M_loss/best_model.pt'))
#checkpoint=torch.load('results/M_6input_M_loss/best_model.pt')
#model.load_state_dict(checkpoint['model_state_dict'])

model.vit.base_model.model.visual_projection.weight.requires_grad=True
model.text_encoder.base_model.model.text_projection.weight.requires_grad=True

optimizer = optim.AdamW(model.parameters(), lr=1e-4,weight_decay=0.05)
#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#current_lr = checkpoint['learning_rate']

#for param_group in optimizer.param_groups:
#    param_group['lr'] = current_lr

criterion = Loss(reduction_type='mean',mse_weight=1,giou_weight=1)

train_loss=[]
validation_loss=[]
train_AIoU=[]
validation_AIoU=[]

epoch_image_pred=[]

early_stopping = EarlyStopping(patience=2, verbose=True, path='/cfs/home/u021542/final_model/4_linear_layers/results/4_LL/best_model_unc.pt')

train_val_path=r'/cfs/home/u021542/final_model/4_linear_layers/results/4_LL/train_val_results_unc.txt'

train_pred=r'/cfs/home/u021542/final_model/4_linear_layers/results/4_LL/draw/train_pred_unc.json'
val_pred=r'/cfs/home/u021542/final_model/4_linear_layers/results/4_LL/draw/val_pred_unc.json'
    
for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    model.train()

    avg_loss, avg_t_IoU, t_img_batch_pred = train_one_epoch()
    
    model.eval()
    
    avg_vloss, avg_v_iou, v_img_batch_pred = val_one_epoch()

    if epoch==0 and not os.path.exists(train_pred):
        #only the first element of the first 50th batches
        write_to_json(train_pred,t_img_batch_pred[:50])
        write_to_json(val_pred,v_img_batch_pred[:50])


    print('AIoU train {} valid {}'.format(avg_t_IoU, avg_v_iou))
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))



    epoch_number += 1
    
    # early_stopping needs the validation loss to check if it has decresed, 
    # and if it has, it will make a checkpoint of the current model
    early_stopping(avg_v_iou, model)

    write_to_file(train_val_path, f'{avg_t_IoU} {avg_v_iou} {avg_loss} {avg_vloss.item()}','Train_AIoU Validation_AIoU')
        
    if early_stopping.early_stop:
        print("Early stopping")
        break

write_to_json(train_pred,t_img_batch_pred[:50])
write_to_json(val_pred,v_img_batch_pred[:50])

#save_results(train_loss,validation_loss,train_AIoU,validation_AIoU)

#draw_bbox(epoch_image_pred)
