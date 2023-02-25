#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import argparse
import numpy as np
import torch
import cv2
from tqdm import tqdm

from utils import set_dataset, dataset_calculator
from data.dataloader import AnyDataset
from extraction_of_CoG import ex_CoG


### parserに関する処理 ###
parser = argparse.ArgumentParser() # インスタンスの生成
parser.add_argument('-clstm', '--convlstm', help='use convlstm as base cell', action='store_true')
parser.add_argument('-cgru', '--convgru', help='use convgru as base cell', action='store_true')
parser.add_argument('-model', default='XXX.pth.tar', help='Specify the file in which the model is stored')
parser.add_argument('--batch_size', default=1, type=int, help='mini-batch size')
args = parser.parse_args() # 引数の解析

### 再現性の担保（シード値の設定） ###
random_seed = 1996
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.device_count() > 1:
    torch.cuda.manual_seed_all(random_seed) # GPUが複数ある場合の処理
else:
    torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

### Datasetの準備 ###
validFolder = AnyDataset(is_train=False, root='./data/', is_augmentation=False)
validLoader = torch.utils.data.DataLoader(validFolder, batch_size=1, shuffle=False)

ds_calc = dataset_calculator()

seq_num = 'seq2_49'
contact_path = 'C:/Users/eugene/Desktop/research_eugene/predict/contact_only_64/' # + seq_num + '/'
save_path = 'C:/Users/eugene/Desktop/research_eugene/predict/output/'

def predict_traj():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # GPUが存在する場合はdeviceとする

    # for i in range(int(len(os.listdir('./data/' + set_dataset() + '/fixed/thermal/val/')))):
    for i in range(int(len(os.listdir(contact_path)))):
        start, end = ds_calc.calc_start_end(i+1)
        point_of_thermal_source = []
    
        t = tqdm(validLoader, leave=False, total=len(validLoader)) # プログレスバーを表示するための仕組み
        for j, (idx, targetVar, inputVar, _, _) in enumerate(t):
            if j < start:
                continue
            if j == end:
                break

            inputs = inputVar.to(device)
            # label = targetVar.to(device)
            
            x_pred, y_pred = ex_CoG(inputs, i, j-start, True)

            if x_pred != None and y_pred != None:
                point_of_thermal_source.append((x_pred, y_pred)) # 熱源推定できているときのみ処理

        ### traj output ###
        traj = np.zeros((64, 64), np.uint8)
        # traj = np.zeros((360, 360), np.uint8)
        pts = np.array(point_of_thermal_source, np.int32)
        # pts = np.unique(pts, axis=0)
        pts = pts.reshape((-1, 1, 2))
        traj = cv2.polylines(traj, [pts], False, (255), thickness=1)
        print('length: ', np.sum(traj==255), '/', traj.size, sep='') # lengthを計算
        cv2.imwrite(save_path + 'pred_img' + str(i+1) + '.jpg', traj, [cv2.IMWRITE_JPEG_QUALITY, 100])
        # cv2.imwrite('./output/CoG_only/pred_img' + str(i+1) + '.png', traj)


if __name__ == "__main__":
    predict_traj()