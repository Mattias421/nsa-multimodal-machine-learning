from sklearn.decomposition import PCA
import os
import tqdm
import pprint
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
#import seaborn as sn
#import matplotlib.pyplot as plt
from utils import utils
from utils import nsa
import torchvision.transforms as transforms
from utils.data_transforms import Unit, Resample
from utils.dataset import MMFit, SequentialStridedSampler
from torch.utils.data import RandomSampler, ConcatDataset
from model.conv_ae import ConvAutoencoder
from model.multimodal_ae import MultimodalAutoencoder
from model.multimodal_ar import MultimodalFcClassifier
from sklearn.metrics import confusion_matrix

"""
evaluate an mm-fit model written by David Stromback, modified by Mattias Cross for fault-tolerance. Such lines have "-MC" commented
"""
################
# Instantiate model
################
args = utils.parse_args()
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(vars(args))
torch.backends.cudnn.benchmark = True

ACTIONS = ['squats', 'lunges', 'bicep_curls', 'situps', 'pushups', 'tricep_extensions', 'dumbbell_rows',
           'jumping_jacks', 'dumbbell_shoulder_press', 'lateral_shoulder_raises', 'non_activity']
TRAIN_W_IDs = []
VAL_W_IDs = []
if args.unseen_test_set:
    TEST_W_IDs = ['00']#, '05', '12', '13', '20']
else:
    TEST_W_IDs = ['09']#, '10', '11']
# All modalities available in MM-Fit
MODALITIES = ['sw_l_acc', 'sw_l_gyr', 'sw_l_hr', 'sw_r_acc', 'sw_r_gyr', 'sw_r_hr', 'sp_l_acc', 'sp_l_gyr',
              'sp_l_mag', 'sp_r_acc', 'sp_r_gyr', 'sp_r_mag', 'eb_l_acc', 'eb_l_gyr', 'pose_2d', 'pose_3d']
# We use a subset of all modalities in this demo.
MODALITIES_SUBSET = ['sw_l_acc', 'sw_l_gyr', 'sw_r_acc', 'sw_r_gyr', 'sp_r_acc', 'sp_r_gyr', 'eb_l_acc', 'eb_l_gyr',
                     'pose_3d']

#currently manually shuting off modalities (credit to David)
ZERO_OUT_MODALITIES = args.zero_out_modalities

NOISY_MODALITIES = args.noisy_modalities #modalities to corrupt - MC
NOISY_FILES = []

IMMUNE_SYSTEM = nsa.generateNI() #CREATE SET OF IMMUNE SYSTEMS - MC

exp_name = args.name
output_path = args.output
if not os.path.exists(output_path):
    os.makedirs(output_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

window_stride = int(args.window_stride * args.skeleton_sampling_rate)
skeleton_window_length = int(args.window_length * args.skeleton_sampling_rate)
sensor_window_length = int(args.window_length * args.target_sensor_sampling_rate)

# Set model training hyperparameters
num_epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.lr

data_transforms = {
        'skeleton': transforms.Compose([
            Unit()
        ]),
        'sensor': transforms.Compose([
            Resample(target_length=sensor_window_length)
        ])
    }

criterion = nn.CrossEntropyLoss()

df = pd.DataFrame(columns=['Epoch', 'Batch', 'Type', 'Loss', 'Accuracy'])
cur_index = 0

#
# NSA DETECTION OCCURS HERE
#

train_datasets, val_datasets, test_datasets = [], [], []
for w_id in TRAIN_W_IDs + VAL_W_IDs + TEST_W_IDs:
    #print(w_id)
    modality_filepaths = {}
    workout_path = os.path.join(args.data, 'w' + w_id)
    files = os.listdir(workout_path)
    label_path = None
    for file in files:
        if 'labels' in file:
            label_path = os.path.join(workout_path, file)
            continue
        for modality_type in MODALITIES_SUBSET: 
            if modality_type in file and modality_type == 'pose_3d':
                modality_filepaths[modality_type] = os.path.join(workout_path, file)
            elif modality_type in file:
                data = np.load(os.path.join(workout_path, file))
  
                #-MC
                if modality_type in NOISY_MODALITIES:
                    #add noise to file
                    data[:, (2, 3, 4)] += np.random.normal(scale = args.noise_scale, size = data[:, (2, 3, 4)].shape)
                    np.save(os.path.join(workout_path, "noisy_" + file), data)
                    NOISY_FILES.append(os.path.join(workout_path, "noisy_" + file))
             
                if nsa.detect(IMMUNE_SYSTEM[modality_type],data,args.m):
                     #noise detected and zeroed out
                     print("NOISE DETECTED IN ",modality_type)
                     modality_filepaths[modality_type] = ""
                     
                elif modality_type in NOISY_MODALITIES:
                    #noise goes through
                    modality_filepaths[modality_type] = os.path.join(workout_path, "noisy_" + file)
                    
                else:
                    #clean data goes through
                    modality_filepaths[modality_type] = os.path.join(workout_path, file)
                
            
    if label_path is None:
        raise Exception('Error: Label file not found for workout {}.'.format(w_id))

    if w_id in TRAIN_W_IDs:
        train_datasets.append(MMFit(modality_filepaths, label_path, args.window_length, skeleton_window_length,
                                    sensor_window_length, skeleton_transform=data_transforms['skeleton'],
                                    sensor_transform=data_transforms['sensor']))
    elif w_id in VAL_W_IDs:
        val_datasets.append(MMFit(modality_filepaths, label_path, args.window_length, skeleton_window_length,
                                  sensor_window_length, skeleton_transform=data_transforms['skeleton'],
                                  sensor_transform=data_transforms['sensor']))
    elif w_id in TEST_W_IDs:
        test_datasets.append(MMFit(modality_filepaths, label_path, args.window_length, skeleton_window_length,
                                   sensor_window_length, skeleton_transform=data_transforms['skeleton'],
                                   sensor_transform=data_transforms['sensor']))
    else:
        raise Exception('Error: Workout {} not assigned to train, test, or val datasets'.format(w_id))


test_dataset = ConcatDataset(test_datasets)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                          sampler=SequentialStridedSampler(test_dataset, window_stride), pin_memory=True)

################
# Evaluation
################

# Test best model
model = torch.load("output/test experiment 5 gpu_e0_best_MODEL.pth") #load model, the one trained by MC was named: "output/test experiment 5 gpu_e0_best_MODEL.pth"
model.eval()

#build confusion matrix https://christianbernecker.medium.com/how-to-create-a-confusion-matrix-in-pytorch-38d06a7f04b7
y_pred = []
y_true = []
with torch.no_grad():
    total, correct, total_loss = 0, 0, 0
    
    with tqdm.tqdm(total=len(test_loader)) as pbar_test:
        for i, (modalities, labels, reps) in enumerate(test_loader):

            for modality, data in modalities.items():
                modalities[modality] = data.to(device, non_blocking=True)

            labels = labels.to(device, non_blocking=True)
            reps = reps.to(device, non_blocking=True)

            outputs = model(modalities['pose_3d'],
                            modalities['eb_l_acc'], modalities['eb_l_gyr'],
                            modalities['sp_r_acc'], modalities['sp_r_gyr'],
                            modalities['sw_l_acc'], modalities['sw_l_gyr'],
                            modalities['sw_r_acc'], modalities['sw_r_gyr'])
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            total_loss_avg = total_loss / ((i + 1) * batch_size)
            total += labels.size(0)

            _, predicted = torch.max(outputs, dim=1)
            batch_correct = (predicted == labels).sum().item()
            correct += batch_correct
            acc = correct / total
            batch_acc = batch_correct / labels.size(0)

            #update confusion matrix
            output = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
            y_pred.extend(output) # Save Prediction
            labels = labels.data.cpu().numpy()
            y_true.extend(labels) # Save Truth

            pbar_test.update(1)
            pbar_test.set_description('Test: Accuracy: {:.4f}, Loss: {:.4f}'.format(acc, total_loss_avg))

# Build confusion matrix

cf_matrix = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in ACTIONS],
                     columns = [i for i in ACTIONS])

df_cm.to_csv(os.path.join(output_path, exp_name + '_CM.csv'), index=False)

for n in NOISY_FILES:
  #remove noisy files after evaluation is done
  os.remove(n)

os._exit(int(acc * 100)) #return accuracy



