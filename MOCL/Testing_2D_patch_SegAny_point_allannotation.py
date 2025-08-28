import argparse
import os, sys
import pandas as pd

sys.path.append("/Data/DoDNet/")

import glob
import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import pickle
import cv2
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

#from scipy.ndimage import morphology
from scipy.ndimage import generate_binary_structure, binary_erosion, distance_transform_edt
from matplotlib import cm

import skimage

import os.path as osp
# from MOTSDataset_2D_Patch_normal import MOTSDataSet, MOTSValDataSet, my_collate
from MOTSDataset_2D_Patch_supervise_csv import MOTSValDataSet as MOTSValDataSet_joint

#from unet2D_ns import UNet2D as UNet2D_ns

import random
import timeit
from tensorboardX import SummaryWriter
import loss_functions.loss_2D as loss

from sklearn import metrics
from math import ceil

from engine import Engine
# from apex import amp
# from apex.parallel import convert_syncbn_model
#from focalloss import FocalLoss2dff
from sklearn.metrics import f1_score, confusion_matrix

start = timeit.default_timer()
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from util.image_pool import ImagePool

from segment_anything import build_sam, SamPredictor, sam_model_registry

torch.cuda.set_device('cuda:0')

from read_roi import read_roi_file
from read_roi import read_roi_zip

def one_hot_2D(targets,C = 2):
    targets_extend=targets.clone()
    targets_extend.unsqueeze_(1) # convert to Nx1xHxW
    one_hot = torch.cuda.FloatTensor(targets_extend.size(0), C, targets_extend.size(2), targets_extend.size(3)).zero_()
    one_hot.scatter_(1, targets_extend, 1)
    return one_hot


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_arguments():

    parser = argparse.ArgumentParser(description="DeepLabV3")

    parser.add_argument("--valset_dir", type=str, default='/content/drive/MyDrive/Doctor_ann/data_list.csv')
    # parser.add_argument("--valset_dir", type=str, default='/Data2/KI_data_validationset_patch/')
    parser.add_argument("--snapshot_dir", type=str, default='snapshots_2D/Segany_point_0525_podocyte/')
    parser.add_argument("--reload_path", type=str, default='snapshots_2D/fold1_with_white_scale_segany_point_0406_test/MOTS_DynConv_fold1_with_white_scale_segany_0406_e100.pth')
    parser.add_argument("--best_epoch", type=int, default=100)
    # parser.add_argument("--validsetname", type=str, default='scale')
    parser.add_argument("--validsetname", type=str, default='normal')
    #parser.add_argument("--valset_dir", type=str, default='/Data2/Demo_KI_data_train_patch_with_white')
    parser.add_argument("--train_list", type=str, default='list/MOTS/MOTS_train.txt')
    parser.add_argument("--val_list", type=str, default='list/MOTS/xx.txt')
    parser.add_argument("--edge_weight", type=float, default=1.2)
    # parser.add_argument("--snapshot_dir", type=str, default='1027results/fold1_with_white_Unet2D_scaleid3_fullydata_1027')
    parser.add_argument("--reload_from_checkpoint", type=str2bool, default=True)
    parser.add_argument("--input_size", type=str, default='256,256')
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument("--FP16", type=str2bool, default=False)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--itrs_each_epoch", type=int, default=250)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--val_pred_every", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--weight_std", type=str2bool, default=True)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--power", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--ignore_label", type=int, default=255)
    parser.add_argument("--is_training", action="store_true")
    parser.add_argument("--random_mirror", type=str2bool, default=True)
    parser.add_argument("--random_scale", type=str2bool, default=True)
    parser.add_argument("--random_seed", type=int, default=1234)
    parser.add_argument("--gpu", type=str, default='None')
    return parser

def count_score_only_two(preds, labels, rmin, rmax, cmin, cmax):

    Val_F1 = 0
    Val_DICE = 0
    Val_TPR = 0
    Val_PPV = 0
    cnt = 0

    for ki in range(len(preds)):
        cnt += 1
        pred = preds[ki,:,rmin[ki]:rmax[ki],cmin[ki]:cmax[ki]]
        label = labels[ki,:,rmin[ki]:rmax[ki],cmin[ki]:cmax[ki]]

        Val_DICE += dice_score(pred, label)
        #preds1 = preds[:,1,...].detach().view(-1).cpu().numpy()
        preds1 = pred[1,...].flatten().detach().cpu().numpy()
        #labels1 = labels[:,1,...].view(-1).cpu().numpy()
        labels1 = label[1,...].detach().flatten().detach().cpu().numpy()

        Val_F1 += f1_score(preds1, labels1, average='macro')

    return Val_F1/cnt, Val_DICE/cnt, 0., 0.

def surfd(input1, input2, sampling=1, connectivity=1):
    # input_1 = np.atleast_1d(input1.astype(bool))
    # input_2 = np.atleast_1d(input2.astype(bool))

    conn = generate_binary_structure(input1.ndim, connectivity)

    S = input1 - binary_erosion(input1, conn)
    Sprime = input2 - binary_erosion(input2, conn)

    S = np.atleast_1d(S.astype(bool))
    Sprime = np.atleast_1d(Sprime.astype(bool))


    dta = distance_transform_edt(~S, sampling)
    dtb = distance_transform_edt(~Sprime, sampling)

    sds = np.concatenate([np.ravel(dta[Sprime != 0]), np.ravel(dtb[S != 0])])

    return np.max(sds), np.mean(sds)


def count_score(preds, labels, rmin, rmax, cmin, cmax):

    Val_F1 = 0
    Val_DICE = 0
    Val_HD = 0
    Val_MSD = 0
    cnt = 0

    for ki in range(len(preds)):
        cnt += 1
        pred = preds[ki,:,rmin[ki]:rmax[ki],cmin[ki]:cmax[ki]]
        label = labels[ki,:,rmin[ki]:rmax[ki],cmin[ki]:cmax[ki]]

        Val_DICE += dice_score(pred, label)
        #preds1 = preds[:,1,...].detach().view(-1).cpu().numpy()
        preds0 = pred[1, ...].detach().cpu().numpy()
        labels0 = label[1, ...].detach().detach().cpu().numpy()

        preds1 = pred[1,...].flatten().detach().cpu().numpy()
        #labels1 = labels[:,1,...].view(-1).cpu().numpy()
        labels1 = label[1,...].detach().flatten().detach().cpu().numpy()

        try:
            hausdorff, meansurfaceDistance = surfd(preds0, labels0)
            Val_HD += hausdorff
            Val_MSD += meansurfaceDistance

            Val_F1 += f1_score(preds1, labels1, average='macro')

        except:
            Val_DICE += 1.
            Val_F1 += 1.
            Val_HD += 0.
            Val_MSD += 0.

    return Val_F1/cnt, Val_DICE/cnt, Val_HD/cnt, Val_MSD/cnt

def dice_score(preds, labels):  # on GPU
    assert preds.shape[0] == labels.shape[0], "predict & target batch size don't match"
    predict = preds.contiguous().view(preds.shape[0], -1)
    target = labels.contiguous().view(labels.shape[0], -1)

    num = torch.sum(torch.mul(predict, target), dim=1)
    den = torch.sum(predict, dim=1) + torch.sum(target, dim=1) + 1e-5

    dice = 2 * num / den

    return dice.mean()

def mask_to_box(tensor):
    tensor = tensor.permute([0,2,3,1]).cpu().numpy()
    rmin = np.zeros((4))
    rmax = np.zeros((4))
    cmin = np.zeros((4))
    cmax = np.zeros((4))
    for ki in range(len(tensor)):
        rows = np.any(tensor[ki], axis=1)
        cols = np.any(tensor[ki], axis=0)

        rmin[ki], rmax[ki] = np.where(rows)[0][[0, -1]]
        cmin[ki], cmax[ki] = np.where(cols)[0][[0, -1]]

    # plt.imshow(tensor[0,int(rmin[0]):int(rmax[0]),int(cmin[0]):int(cmax[0]),:])
    return rmin.astype(np.uint32), rmax.astype(np.uint32), cmin.astype(np.uint32), cmax.astype(np.uint32)


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))



import numpy as np
import matplotlib.pyplot as plt
import cv2
import natsort
import glob
import random

random_seed = 2023
random.seed(random_seed)
np.random.seed(random_seed)

def randomseed_perComp(mask_img, min_comp_area=1, mode='random', erode_size=3, comp_with_seed_rate=1.0): #        kernel = np.ones((3, 3), np.uint8)
   """
   :param mask_img: Binary mask. nparray. Size H x W, intensity range [0,155].
   :param min_comp_area: selected component must contain more than min_comp_area pixels. int.
   :param mode: 'random' or 'center'
   :param erode_size: param for erosion filter. int.
   :param comp_with_seed_rate: Select comp_with_seed_rate * 100 percent valid components. Generate one seed for each component. Range: [0,1]. float.
   """

   # Erode and binarize
   #img_ori = np.copy(mask_img)
   if erode_size > 0:
       kernel = np.ones((erode_size, erode_size), np.uint8)
       mask_img = cv2.erode(mask_img, kernel, iterations=1)
       _, mask_img = cv2.threshold(mask_img, 100, 255, cv2.THRESH_BINARY)


   # Connected components
   num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_img)
   large_components_idx = np.where(stats[:, -1] > min_comp_area)[0]

   # Remove the background component
   if 0 in large_components_idx: large_components_idx = np.delete(large_components_idx, np.where(large_components_idx == 0))

   large_components_idx = random.sample(list(large_components_idx), int(len(large_components_idx) * comp_with_seed_rate))


   # Generate a seed for each selected component
   output = []
   for idx in large_components_idx:
       component_mask = np.uint8(labels == idx) * 255
       if mode == 'center':
           output.append(centroids[idx]) #average of the x and y coordinates of all the pixels in the component
           if component_mask[int(centroids[idx][1]),int(centroids[idx][0])] == 0:
               print('center point is out of the component!')
       elif mode == 'random':
           x, y = np.where(component_mask == 255)
           i = np.random.randint(len(x))
           random_pos = [y[i],x[i]]
           output.append(random_pos)
   return output, mask_img


#
# def main():
#    img_dir = r'/home/VANDERBILT/cuic4/Research/Code/exampleData/'
#    for filename in natsort.natsorted(glob.glob(img_dir + 'TEST5.png')):
#        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
#        _, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
#
#        out, eroded_img = randomseed_perComp(img, min_comp_area=0, mode='random', erode_size=22, comp_with_seed_rate=1)
#        # check results:
#        plt.imshow(img)
#        out = np.array(out)
#        plt.scatter(out[:,0], out[:,1], color='red', s=10)
#        plt.show()
#
#        plt.imshow(eroded_img)
#        out = np.array(out)
#        plt.scatter(out[:,0], out[:,1], color='red', s=10)
#        #plt.scatter(out[1][0], out[1][1], color='red', s=10)
#        plt.show()
#
# main()


"""
plt.imshow(img)
plt.scatter(out[0][0], out[0][1], color='red', s=10)
plt.scatter(out[1][0], out[1][1], color='red', s=10)
plt.show()
"""




def main():
    """Create the model and start the training."""
    parser = get_arguments()
    print(parser)

    with Engine(custom_parser=parser) as engine:
        args = parser.parse_args()
        if args.num_gpus > 1:
            torch.cuda.set_device(args.local_rank)

        writer = SummaryWriter(args.snapshot_dir)

        if not args.gpu == 'None':
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

        h, w = map(int, args.input_size.split(','))
        input_size = (h, w)

        cudnn.benchmark = True
        seed = args.random_seed
        if engine.distributed:
            seed = args.local_rank
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # Create model
        criterion = None
        # model = UNet2D_ns(num_classes=args.num_classes, weight_std = False)
        # check_wo_gpu = 0

        sam_checkpoint = "/content/drive/MyDrive/ToXueyuan/checkpoint/sam_vit_h_4b8939.pth"
        device = "cuda:0"
        model_type = "default"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        model = SamPredictor(sam)
        #model = SamPredictor(build_sam(checkpoint="/Data2/HumanKidney/SegmentAnything/checkpoint/sam_vit_h_4b8939.pth"))


        #print(f"total params: \t{sum([np.prod(p.size()) for p in model.parameters()])}")

        # if not check_wo_gpu:
        #device = torch.device('cuda:{}'.format(args.local_rank))
        # device = torch.device('cuda:0')
        # model.to(device)

        #optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.99, nesterov=True)

        # if not check_wo_gpu:
        #     if args.FP16:
        #         print("Note: Using FP16 during training************")
        #         model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        #
        #     if args.num_gpus > 1:
        #         model = engine.data_parallel(model)
        #
        # # load checkpoint...
        # if args.reload_from_checkpoint:
        #     print('loading from checkpoint: {}'.format(args.reload_path))
        #     if os.path.exists(args.reload_path):
        #         if args.FP16:
        #             checkpoint = torch.load(args.reload_path, map_location=torch.device('cpu'))
        #             model.load_state_dict(checkpoint['model'])
        #             optimizer.load_state_dict(checkpoint['optimizer'])
        #             amp.load_state_dict(checkpoint['amp'])
        #         else:
        #             model.load_state_dict(torch.load(args.reload_path, map_location=torch.device('cpu')))
        #     else:
        #         print('File not exists in the reload path: {}'.format(args.reload_path))


        if not os.path.exists(args.snapshot_dir):
            os.makedirs(args.snapshot_dir)

        edge_weight = args.edge_weight

        num_worker = 0


        valloader = DataLoader(
            MOTSValDataSet_joint(args.valset_dir, args.val_list, max_iters=args.itrs_each_epoch * args.batch_size,
                           crop_size=input_size, scale=args.random_scale, mirror=args.random_mirror,
                           edge_weight=edge_weight),batch_size=1,shuffle=False,num_workers=num_worker)

        all_tr_loss = []
        all_va_loss = []
        train_loss_MA = None
        val_loss_MA = None

        val_best_loss = 999999
        batch_size = 1
        # for epoch in range(0,args.num_epochs):

        #model.load_state_dict(torch.load(args.reload_path, map_location=torch.device('cpu')))

        #model.eval()
        task0_pool_image = ImagePool(8)
        task0_pool_mask = ImagePool(8)
        task0_scale = []
        task0_name = []
        task1_pool_image = ImagePool(8)
        task1_pool_mask = ImagePool(8)
        task1_scale = []
        task1_name = []
        task2_pool_image = ImagePool(8)
        task2_pool_mask = ImagePool(8)
        task2_scale = []
        task2_name = []
        task3_pool_image = ImagePool(8)
        task3_pool_mask = ImagePool(8)
        task3_scale = []
        task3_name = []
        task4_pool_image = ImagePool(8)
        task4_pool_mask = ImagePool(8)
        task4_scale = []
        task4_name = []
        task5_pool_image = ImagePool(8)
        task5_pool_mask = ImagePool(8)
        task5_scale = []
        task5_name = []

        val_loss = np.zeros((6))
        val_F1 = np.zeros((6))
        val_Dice = np.zeros((6))
        val_HD = np.zeros((6))
        val_MSD = np.zeros((6))
        cnt = np.zeros((6))

        single_df_0 = pd.DataFrame(columns=['name', 'F1', 'Dice', 'HD', 'MSD'])
        single_df_1 = pd.DataFrame(columns=['name', 'F1', 'Dice', 'HD', 'MSD'])
        single_df_2 = pd.DataFrame(columns=['name', 'F1', 'Dice', 'HD', 'MSD'])
        single_df_3 = pd.DataFrame(columns=['name', 'F1', 'Dice', 'HD', 'MSD'])
        single_df_4 = pd.DataFrame(columns=['name', 'F1', 'Dice', 'HD', 'MSD'])
        single_df_5 = pd.DataFrame(columns=['name', 'F1', 'Dice', 'HD', 'MSD'])

        with torch.no_grad():
            for iter, batch in enumerate(valloader):

                'dataloader'
                imgs = batch[0].cuda()
                lbls = batch[1].cuda()
                wt = batch[2].cuda().float()
                volumeName = batch[3]
                t_ids = batch[4].cuda()
                s_ids = batch[5]

                for ki in range(len(imgs)):
                    now_task = t_ids[ki]

                    # if now_task != 0:
                    #     continue

                    if now_task == 0:
                        task0_pool_image.add(imgs[ki].unsqueeze(0))
                        task0_pool_mask.add(lbls[ki].unsqueeze(0))
                        task0_scale.append((s_ids[ki]))
                        task0_name.append((volumeName[ki]))
                    elif now_task == 1:
                        task1_pool_image.add(imgs[ki].unsqueeze(0))
                        task1_pool_mask.add(lbls[ki].unsqueeze(0))
                        task1_scale.append((s_ids[ki]))
                        task1_name.append((volumeName[ki]))
                    elif now_task == 2:
                        task2_pool_image.add(imgs[ki].unsqueeze(0))
                        task2_pool_mask.add(lbls[ki].unsqueeze(0))
                        task2_scale.append((s_ids[ki]))
                        task2_name.append((volumeName[ki]))
                    elif now_task == 3:
                        task3_pool_image.add(imgs[ki].unsqueeze(0))
                        task3_pool_mask.add(lbls[ki].unsqueeze(0))
                        task3_scale.append((s_ids[ki]))
                        task3_name.append((volumeName[ki]))
                    elif now_task == 4:
                        task4_pool_image.add(imgs[ki].unsqueeze(0))
                        task4_pool_mask.add(lbls[ki].unsqueeze(0))
                        task4_scale.append((s_ids[ki]))
                        task4_name.append((volumeName[ki]))
                    elif now_task == 5:
                        task5_pool_image.add(imgs[ki].unsqueeze(0))
                        task5_pool_mask.add(lbls[ki].unsqueeze(0))
                        task5_scale.append((s_ids[ki]))
                        task5_name.append((volumeName[ki]))


                # = os.path.join(args.snapshot_dir.replace('snapshots_2D/fold1_with_white','/Data2/cell_segmentation/Expert_%s' % (args.validsetname)), str(args.best_epoch))
                output_folder = os.path.join('/content/drive/MyDrive/Doctor_ann/Expert_%s' % (args.validsetname), str(args.best_epoch))
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                #optimizer.zero_grad()

                if task0_pool_image.num_imgs >= batch_size:
                    images = task0_pool_image.query(batch_size)
                    labels = task0_pool_mask.query(batch_size)
                    now_task = torch.tensor(0)
                    scales = torch.ones(batch_size).cuda()
                    filename = []
                    for bi in range(len(scales)):
                        scales[int(len(scales) - 1 - bi)] = task0_scale.pop(0)
                        filename.append(task0_name.pop(0))

                    #preds, _ = model(images, torch.ones(batch_size).cuda() * 0, scales)


                    'SAM'
                    now_preds = torch.zeros((images.shape[0],images.shape[2],images.shape[3]))

                    input_point_list = []
                    input_label_list = []

                    for ni in range(images.shape[0]):
                        now_img = images[ni]

                        #model.set_image(now_img)
                        model.set_image((now_img.permute([1,2,0]).cpu().numpy() * 255).astype(np.uint8))

                        'get out from roi regions'

                        'get out from roi regions'
                        roi_root = '/content/drive/MyDrive/Docter_annotation_roi/1_2_mesangial_roi/%s' % (filename[ni].split('-')[-1].replace('.png','.zip'))
                        if not os.path.exists(roi_root):
                            roi_root = '/content/drive/MyDrive/Docter_annotation_roi/1_2_mesangial_roi/%s' % (
                                filename[ni].split('-')[-1].replace('.png', '.roi'))
                            rois = read_roi_file(roi_root)

                            input_point = np.zeros((len(rois), 2)).astype(np.int)
                            ann_key = list(rois.keys())

                            for ri in range(len(rois)):
                                input_point[ri, 0] = rois[ann_key[ri]]['x'][0]
                                input_point[ri, 1] = rois[ann_key[ri]]['y'][0]

                            input_label = np.ones((len(input_point)))


                        else:
                            rois = read_roi_zip(roi_root)

                            input_point = np.zeros((len(rois), 2)).astype(np.int)
                            ann_key = list(rois.keys())

                            for ri in range(len(rois)):
                                input_point[ri, 0] = rois[ann_key[-1]]['x'][ri]
                                input_point[ri, 1] = rois[ann_key[-1]]['y'][ri]

                            input_label = np.ones((len(input_point)))


                        if len(input_point) > 0:
                            masks = np.zeros((1, images.shape[2], images.shape[3]))
                            for pp in range(len(input_point)):

                                now_input_point = input_point[pp:pp+1]
                                now_input_label = input_label[pp:pp+1]

                                now_mask, scores, logits = model.predict(
                                    point_coords=now_input_point,
                                    point_labels=now_input_label,
                                    multimask_output=False,
                                )

                                masks += now_mask

                            masks[masks>1] = 1

                        else:
                            input_point = None
                            input_label = None

                            print('error_%s' % (filename[ni]))
                            masks, scores, logits = model.predict(
                                point_coords=None,
                                point_labels=None,
                                multimask_output=False,
                            )

                        input_point_list.append((input_point))
                        input_label_list.append((input_label))

                        now_preds[ni] = torch.from_numpy(masks[0])

                    #now_preds = preds[:,1,...] > preds[:,0,...]
                    now_preds_onehot = one_hot_2D(now_preds.long().cuda())

                    labels_onehot = one_hot_2D(labels.long())

                    rmin, rmax, cmin, cmax = mask_to_box(images)

                    for pi in range(len(images)):
                        prediction = now_preds[pi]
                        out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
                        img = (out_image - out_image.min())/ (out_image.max() - out_image.min())
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
                                   img)
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
                                   labels[pi, ...].detach().cpu().numpy(), cmap = cm.gray)
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_preds_%s.png' %(now_task.item())),
                                   prediction.detach().cpu().numpy(), cmap = cm.gray)

                        plt.figure(figsize=(10, 10))
                        plt.imshow(img,alpha=0.5)
                        show_mask(prediction.detach().cpu().numpy(), plt.gca())

                        if not (input_point_list[pi] is None):
                            show_points(input_point_list[pi], input_label_list[pi], plt.gca())

                        plt.axis('off')
                        plt.savefig(os.path.join(output_folder, filename[pi] + '_segany_%s.png' % (now_task.item())),bbox_inches='tight',pad_inches=0)
                        plt.close()


                        'SAM end'

                        F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0), labels_onehot[pi].unsqueeze(0),
                                                         rmin, rmax, cmin, cmax)
                        row = len(single_df_0)
                        single_df_0.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]

                        val_F1[0] += F1
                        val_Dice[0] += DICE
                        val_HD[0] += HD
                        val_MSD[0] += MSD
                        cnt[0] += 1

                if task1_pool_image.num_imgs >= batch_size:
                    images = task1_pool_image.query(batch_size)
                    labels = task1_pool_mask.query(batch_size)
                    scales = torch.ones(batch_size).cuda()
                    filename = []
                    for bi in range(len(scales)):
                        scales[int(len(scales) - 1 - bi)] = task1_scale.pop(0)
                        filename.append(task1_name.pop(0))

                    #preds, _ = model(images, torch.ones(batch_size).cuda() * 1, scales)
                    now_task = torch.tensor(1)

                    'SAM'
                    now_preds = torch.zeros((images.shape[0], images.shape[2], images.shape[3]))

                    input_point_list = []
                    input_label_list = []

                    for ni in range(images.shape[0]):
                        now_img = images[ni]

                        # model.set_image(now_img)
                        model.set_image((now_img.permute([1, 2, 0]).cpu().numpy() * 255).astype(np.uint8))

                        'get out from roi regions'
                        roi_root = '/content/drive/MyDrive/Docter_annotation_roi/1_2_mesangial_roi/%s' % (filename[ni].split('-')[-1].replace('.png','.zip'))
                        if not os.path.exists(roi_root):
                            roi_root = '/content/drive/MyDrive/Docter_annotation_roi/1_2_mesangial_roi/%s' % (
                                filename[ni].split('-')[-1].replace('.png', '.roi'))

                            if not os.path.exists(roi_root):
                                break

                            rois = read_roi_file(roi_root)

                            input_point = np.zeros((len(rois), 2)).astype(np.int)
                            ann_key = list(rois.keys())

                            for ri in range(len(rois)):
                                input_point[ri, 0] = rois[ann_key[ri]]['x'][0]
                                input_point[ri, 1] = rois[ann_key[ri]]['y'][0]

                            input_label = np.ones((len(input_point)))


                        else:
                            rois = read_roi_zip(roi_root)

                            input_point = np.zeros((len(rois), 2)).astype(np.int)
                            ann_key = list(rois.keys())

                            for ri in range(len(rois)):
                                input_point[ri, 0] = rois[ann_key[-1]]['x'][ri]
                                input_point[ri, 1] = rois[ann_key[-1]]['y'][ri]

                            input_label = np.ones((len(input_point)))

                        if len(input_point) > 0:
                            masks = np.zeros((1, images.shape[2], images.shape[3]))
                            for pp in range(len(input_point)):
                                now_input_point = input_point[pp:pp + 1]
                                now_input_label = input_label[pp:pp + 1]

                                now_mask, scores, logits = model.predict(
                                    point_coords=now_input_point,
                                    point_labels=now_input_label,
                                    multimask_output=False,
                                )

                                masks += now_mask

                            masks[masks > 1] = 1

                        else:
                            input_point = None
                            input_label = None

                            print('error_%s' % (filename[ni]))
                            masks, scores, logits = model.predict(
                                point_coords=None,
                                point_labels=None,
                                multimask_output=False,
                            )

                        input_point_list.append((input_point))
                        input_label_list.append((input_label))

                        now_preds[ni] = torch.from_numpy(masks[0])

                    # now_preds = preds[:,1,...] > preds[:,0,...]
                    now_preds_onehot = one_hot_2D(now_preds.long().cuda())

                    labels_onehot = one_hot_2D(labels.long())

                    rmin, rmax, cmin, cmax = mask_to_box(images)

                    for pi in range(len(images)):
                        prediction = now_preds[pi]
                        out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
                        img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
                                   img)
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
                                   labels[pi, ...].detach().cpu().numpy(), cmap=cm.gray)
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
                                   prediction.detach().cpu().numpy(), cmap=cm.gray)

                        plt.figure(figsize=(10, 10))
                        plt.imshow(img,alpha=0.5)
                        show_mask(prediction.detach().cpu().numpy(), plt.gca())

                        if not (input_point_list[pi] is None):
                            show_points(input_point_list[pi], input_label_list[pi], plt.gca())

                        plt.axis('off')
                        plt.savefig(os.path.join(output_folder, filename[pi] + '_segany_%s.png' % (now_task.item())),bbox_inches='tight',pad_inches=0)
                        plt.close()

                        'SAM end'

                        F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0), labels_onehot[pi].unsqueeze(0),
                                                         rmin, rmax, cmin, cmax)
                        row = len(single_df_1)
                        single_df_1.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]

                        val_F1[1] += F1
                        val_Dice[1] += DICE
                        val_HD[1] += HD
                        val_MSD[1] += MSD
                        cnt[1] += 1


                if task2_pool_image.num_imgs >= batch_size:
                    images = task2_pool_image.query(batch_size)
                    labels = task2_pool_mask.query(batch_size)
                    scales = torch.ones(batch_size).cuda()
                    filename = []
                    for bi in range(len(scales)):
                        scales[int(len(scales) - 1 - bi)] = task2_scale.pop(0)
                        filename.append(task2_name.pop(0))

                    #preds, _ = model(images, torch.ones(batch_size).cuda() * 2, scales)
                    now_task = torch.tensor(2)

                    'SAM'
                    now_preds = torch.zeros((images.shape[0], images.shape[2], images.shape[3]))

                    input_point_list = []
                    input_label_list = []

                    for ni in range(images.shape[0]):
                        now_img = images[ni]

                        # model.set_image(now_img)
                        model.set_image((now_img.permute([1, 2, 0]).cpu().numpy() * 255).astype(np.uint8))

                        mask_img = (labels[ni].cpu().numpy() * 255).astype(np.uint8)
                        out, _ = randomseed_perComp(mask_img, min_comp_area=1, mode='random', erode_size=10,
                                                    comp_with_seed_rate=1.0)
                        # out = randomseed_perComp((labels[ni].cpu().numpy() * 255).astype(np.uint8), min_comp_area=10, mode='random')

                        if len(out) == 0:
                            out, _ = randomseed_perComp(mask_img, min_comp_area=1, mode='random', erode_size=0,
                                                        comp_with_seed_rate=1.0)

                        input_point = np.array((out))
                        input_label = np.ones((len(out)))

                        if len(input_point) > 0:
                            masks = np.zeros((1, mask_img.shape[0], mask_img.shape[1]))
                            for pp in range(len(input_point)):
                                now_input_point = input_point[pp:pp + 1]
                                now_input_label = input_label[pp:pp + 1]

                                now_mask, scores, logits = model.predict(
                                    point_coords=now_input_point,
                                    point_labels=now_input_label,
                                    multimask_output=False,
                                )

                                masks += now_mask

                            masks[masks > 1] = 1

                        else:
                            input_point = None
                            input_label = None

                            print('error_%s' % (filename[ni]))
                            masks, scores, logits = model.predict(
                                point_coords=None,
                                point_labels=None,
                                multimask_output=False,
                            )

                        input_point_list.append((input_point))
                        input_label_list.append((input_label))

                        now_preds[ni] = torch.from_numpy(masks[0])

                    # now_preds = preds[:,1,...] > preds[:,0,...]
                    now_preds_onehot = one_hot_2D(now_preds.long().cuda())

                    labels_onehot = one_hot_2D(labels.long())

                    rmin, rmax, cmin, cmax = mask_to_box(images)

                    for pi in range(len(images)):
                        prediction = now_preds[pi]
                        out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
                        img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
                                   img)
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
                                   labels[pi, ...].detach().cpu().numpy(), cmap=cm.gray)
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
                                   prediction.detach().cpu().numpy(), cmap=cm.gray)

                        plt.figure(figsize=(10, 10))
                        plt.imshow(img,alpha=0.5)
                        show_mask(prediction.detach().cpu().numpy(), plt.gca())

                        if not (input_point_list[pi] is None):
                            show_points(input_point_list[pi], input_label_list[pi], plt.gca())

                        plt.axis('off')
                        plt.savefig(os.path.join(output_folder, filename[pi] + '_segany_%s.png' % (now_task.item())),bbox_inches='tight',pad_inches=0)
                        plt.close()

                        'SAM end'

                        F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0), labels_onehot[pi].unsqueeze(0),
                                                         rmin, rmax, cmin, cmax)
                        row = len(single_df_2)
                        single_df_2.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]

                        val_F1[2] += F1
                        val_Dice[2] += DICE
                        val_HD[2] += HD
                        val_MSD[2] += MSD
                        cnt[2] += 1


                if task3_pool_image.num_imgs >= batch_size:
                    images = task3_pool_image.query(batch_size)
                    labels = task3_pool_mask.query(batch_size)
                    scales = torch.ones(batch_size).cuda()
                    filename = []
                    for bi in range(len(scales)):
                        scales[int(len(scales) - 1 - bi)] = task3_scale.pop(0)
                        filename.append(task3_name.pop(0))

                    #preds, _ = model(images, torch.ones(batch_size).cuda() * 3, scales)
                    now_task = torch.tensor(3)

                    'SAM'
                    now_preds = torch.zeros((images.shape[0], images.shape[2], images.shape[3]))

                    input_point_list = []
                    input_label_list = []

                    for ni in range(images.shape[0]):
                        now_img = images[ni]

                        # model.set_image(now_img)
                        model.set_image((now_img.permute([1, 2, 0]).cpu().numpy() * 255).astype(np.uint8))

                        mask_img = (labels[ni].cpu().numpy() * 255).astype(np.uint8)
                        out, _ = randomseed_perComp(mask_img, min_comp_area=1, mode='random', erode_size=10,
                                                    comp_with_seed_rate=1.0)
                        # out = randomseed_perComp((labels[ni].cpu().numpy() * 255).astype(np.uint8), min_comp_area=10, mode='random')

                        if len(out) == 0:
                            out, _ = randomseed_perComp(mask_img, min_comp_area=1, mode='random', erode_size=0,
                                                        comp_with_seed_rate=1.0)

                        input_point = np.array((out))
                        input_label = np.ones((len(out)))

                        if len(input_point) > 0:
                            masks = np.zeros((1, mask_img.shape[0], mask_img.shape[1]))
                            for pp in range(len(input_point)):
                                now_input_point = input_point[pp:pp + 1]
                                now_input_label = input_label[pp:pp + 1]

                                now_mask, scores, logits = model.predict(
                                    point_coords=now_input_point,
                                    point_labels=now_input_label,
                                    multimask_output=False,
                                )

                                masks += now_mask

                            masks[masks > 1] = 1

                        else:
                            input_point = None
                            input_label = None

                            print('error_%s' % (filename[ni]))
                            masks, scores, logits = model.predict(
                                point_coords=None,
                                point_labels=None,
                                multimask_output=False,
                            )

                        input_point_list.append((input_point))
                        input_label_list.append((input_label))

                        now_preds[ni] = torch.from_numpy(masks[0])

                    # now_preds = preds[:,1,...] > preds[:,0,...]
                    now_preds_onehot = one_hot_2D(now_preds.long().cuda())

                    labels_onehot = one_hot_2D(labels.long())

                    rmin, rmax, cmin, cmax = mask_to_box(images)

                    for pi in range(len(images)):
                        prediction = now_preds[pi]
                        out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
                        img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
                                   img)
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
                                   labels[pi, ...].detach().cpu().numpy(), cmap=cm.gray)
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
                                   prediction.detach().cpu().numpy(), cmap=cm.gray)

                        plt.figure(figsize=(10, 10))
                        plt.imshow(img,alpha=0.5)
                        show_mask(prediction.detach().cpu().numpy(), plt.gca())

                        if not (input_point_list[pi] is None):
                            show_points(input_point_list[pi], input_label_list[pi], plt.gca())

                        plt.axis('off')
                        plt.savefig(os.path.join(output_folder, filename[pi] + '_segany_%s.png' % (now_task.item())),bbox_inches='tight',pad_inches=0)
                        plt.close()

                        'SAM end'

                        F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0), labels_onehot[pi].unsqueeze(0),
                                                         rmin, rmax, cmin, cmax)
                        row = len(single_df_3)
                        single_df_3.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]

                        val_F1[3] += F1
                        val_Dice[3] += DICE
                        val_HD[3] += HD
                        val_MSD[3] += MSD
                        cnt[3] += 1

                if task4_pool_image.num_imgs >= batch_size:
                    images = task4_pool_image.query(batch_size)
                    labels = task4_pool_mask.query(batch_size)
                    scales = torch.ones(batch_size).cuda()
                    filename = []
                    for bi in range(len(scales)):
                        scales[int(len(scales) - 1 - bi)] = task4_scale.pop(0)
                        filename.append(task4_name.pop(0))

                    # preds, _ = model(images, torch.ones(batch_size).cuda() * 4, scales)
                    now_task = torch.tensor(4)

                    'SAM'
                    now_preds = torch.zeros((images.shape[0], images.shape[2], images.shape[3]))

                    input_point_list = []
                    input_label_list = []

                    for ni in range(images.shape[0]):
                        now_img = images[ni]

                        # model.set_image(now_img)
                        model.set_image((now_img.permute([1, 2, 0]).cpu().numpy() * 255).astype(np.uint8))

                        mask_img = (labels[ni].cpu().numpy() * 255).astype(np.uint8)
                        out, _ = randomseed_perComp(mask_img, min_comp_area=1, mode='random', erode_size=10,
                                                    comp_with_seed_rate=1.0)
                        # out = randomseed_perComp((labels[ni].cpu().numpy() * 255).astype(np.uint8), min_comp_area=10, mode='random')

                        if len(out) == 0:
                            out, _ = randomseed_perComp(mask_img, min_comp_area=1, mode='random', erode_size=0,
                                                        comp_with_seed_rate=1.0)

                        input_point = np.array((out))
                        input_label = np.ones((len(out)))

                        if len(input_point) > 0:
                            masks = np.zeros((1, mask_img.shape[0], mask_img.shape[1]))
                            for pp in range(len(input_point)):
                                now_input_point = input_point[pp:pp + 1]
                                now_input_label = input_label[pp:pp + 1]

                                now_mask, scores, logits = model.predict(
                                    point_coords=now_input_point,
                                    point_labels=now_input_label,
                                    multimask_output=False,
                                )

                                masks += now_mask

                            masks[masks > 1] = 1

                        else:
                            input_point = None
                            input_label = None

                            print('error_%s' % (filename[ni]))
                            masks, scores, logits = model.predict(
                                point_coords=None,
                                point_labels=None,
                                multimask_output=False,
                            )

                        input_point_list.append((input_point))
                        input_label_list.append((input_label))

                        now_preds[ni] = torch.from_numpy(masks[0])

                    # now_preds = preds[:,1,...] > preds[:,0,...]
                    now_preds_onehot = one_hot_2D(now_preds.long().cuda())

                    labels_onehot = one_hot_2D(labels.long())

                    rmin, rmax, cmin, cmax = mask_to_box(images)

                    for pi in range(len(images)):
                        prediction = now_preds[pi]
                        out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
                        img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
                                   img)
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
                                   labels[pi, ...].detach().cpu().numpy(), cmap=cm.gray)
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
                                   prediction.detach().cpu().numpy(), cmap=cm.gray)

                        plt.figure(figsize=(10, 10))
                        plt.imshow(img,alpha=0.5)
                        show_mask(prediction.detach().cpu().numpy(), plt.gca())

                        if not (input_point_list[pi] is None):
                            show_points(input_point_list[pi], input_label_list[pi], plt.gca())

                        plt.axis('off')
                        plt.savefig(os.path.join(output_folder, filename[pi] + '_segany_%s.png' % (now_task.item())),bbox_inches='tight',pad_inches=0)
                        plt.close()

                        'SAM end'

                        F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0), labels_onehot[pi].unsqueeze(0),
                                                         rmin, rmax, cmin, cmax)
                        row = len(single_df_4)
                        single_df_4.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]

                        val_F1[4] += F1
                        val_Dice[4] += DICE
                        val_HD[4] += HD
                        val_MSD[4] += MSD
                        cnt[4] += 1

                if task5_pool_image.num_imgs >= batch_size:
                    images = task5_pool_image.query(batch_size)
                    labels = task5_pool_mask.query(batch_size)
                    scales = torch.ones(batch_size).cuda()
                    filename = []
                    for bi in range(len(scales)):
                        scales[int(len(scales) - 1 - bi)] = task5_scale.pop(0)
                        filename.append(task5_name.pop(0))

                    # preds, _ = model(images, torch.ones(batch_size).cuda() * 5, scales)
                    now_task = torch.tensor(5)

                    'SAM'
                    now_preds = torch.zeros((images.shape[0], images.shape[2], images.shape[3]))

                    input_point_list = []
                    input_label_list = []

                    for ni in range(images.shape[0]):
                        now_img = images[ni]

                        # model.set_image(now_img)
                        model.set_image((now_img.permute([1, 2, 0]).cpu().numpy() * 255).astype(np.uint8))

                        mask_img = (labels[ni].cpu().numpy() * 255).astype(np.uint8)
                        out, _ = randomseed_perComp(mask_img, min_comp_area=1, mode='random', erode_size=10,
                                                    comp_with_seed_rate=1.0)
                        # out = randomseed_perComp((labels[ni].cpu().numpy() * 255).astype(np.uint8), min_comp_area=10, mode='random')

                        if len(out) == 0:
                            out, _ = randomseed_perComp(mask_img, min_comp_area=1, mode='random', erode_size=0,
                                                        comp_with_seed_rate=1.0)

                        input_point = np.array((out))
                        input_label = np.ones((len(out)))

                        if len(input_point) > 0:
                            masks = np.zeros((1, mask_img.shape[0], mask_img.shape[1]))
                            for pp in range(len(input_point)):
                                now_input_point = input_point[pp:pp + 1]
                                now_input_label = input_label[pp:pp + 1]

                                now_mask, scores, logits = model.predict(
                                    point_coords=now_input_point,
                                    point_labels=now_input_label,
                                    multimask_output=False,
                                )

                                masks += now_mask

                            masks[masks > 1] = 1

                        else:
                            input_point = None
                            input_label = None

                            print('error_%s' % (filename[ni]))
                            masks, scores, logits = model.predict(
                                point_coords=None,
                                point_labels=None,
                                multimask_output=False,
                            )

                        input_point_list.append((input_point))
                        input_label_list.append((input_label))

                        now_preds[ni] = torch.from_numpy(masks[0])

                    # now_preds = preds[:,1,...] > preds[:,0,...]
                    now_preds_onehot = one_hot_2D(now_preds.long().cuda())

                    labels_onehot = one_hot_2D(labels.long())

                    rmin, rmax, cmin, cmax = mask_to_box(images)

                    for pi in range(len(images)):
                        prediction = now_preds[pi]
                        out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
                        img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
                                   img)
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
                                   labels[pi, ...].detach().cpu().numpy(), cmap=cm.gray)
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
                                   prediction.detach().cpu().numpy(), cmap=cm.gray)

                        plt.figure(figsize=(10, 10))
                        plt.imshow(img,alpha=0.5)
                        show_mask(prediction.detach().cpu().numpy(), plt.gca())

                        if not (input_point_list[pi] is None):
                            show_points(input_point_list[pi], input_label_list[pi], plt.gca())

                        plt.axis('off')
                        plt.savefig(os.path.join(output_folder, filename[pi] + '_segany_%s.png' % (now_task.item())),bbox_inches='tight',pad_inches=0)
                        plt.close()

                        'SAM end'

                        F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0), labels_onehot[pi].unsqueeze(0),
                                                         rmin, rmax, cmin, cmax)
                        row = len(single_df_5)
                        single_df_5.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]

                        val_F1[5] += F1
                        val_Dice[5] += DICE
                        val_HD[5] += HD
                        val_MSD[5] += MSD
                        cnt[5] += 1


        avg_val_F1 = val_F1 / cnt
        avg_val_Dice = val_Dice / cnt
        avg_val_HD = val_HD / cnt
        avg_val_MSD = val_MSD / cnt

        print('Validate \n 0dt_f1={:.4} 0dt_dsc={:.4} 0dt_hd={:.4} 0dt_msd={:.4}'
              ' \n 1pt_f1={:.4} 1pt_dsc={:.4} 1pt_hd={:.4} 1pt_msd={:.4}\n'
              ' \n 2cps_f1={:.4} 2cps_dsc={:.4} 2cps_hd={:.4} 2cps_msd={:.4}\n'
              ' \n 3tf_f1={:.4} 3tf_dsc={:.4} 3tf_hd={:.4} 3tf_msd={:.4}\n'
              ' \n 4vs_f1={:.4} 4vs_dsc={:.4} 4vs_hd={:.4} 4vs_msd={:.4}\n'
              ' \n 5ptc_f1={:.4} 5ptc_dsc={:.4} 5ptc_hd={:.4} 5ptc_msd={:.4}\n'
              .format(avg_val_F1[0].item(), avg_val_Dice[0].item(), avg_val_HD[0].item(), avg_val_MSD[0].item(),
                      avg_val_F1[1].item(), avg_val_Dice[1].item(), avg_val_HD[1].item(), avg_val_MSD[1].item(),
                      avg_val_F1[2].item(), avg_val_Dice[2].item(), avg_val_HD[2].item(), avg_val_MSD[2].item(),
                      avg_val_F1[3].item(), avg_val_Dice[3].item(), avg_val_HD[3].item(), avg_val_MSD[3].item(),
                      avg_val_F1[4].item(), avg_val_Dice[4].item(), avg_val_HD[4].item(), avg_val_MSD[4].item(),
                      avg_val_F1[5].item(), avg_val_Dice[5].item(), avg_val_HD[5].item(), avg_val_MSD[5].item()))

        df = pd.DataFrame(columns = ['task','F1','Dice','HD','MSD'])
        df.loc[0] = ['0dt', avg_val_F1[0].item(), avg_val_Dice[0].item(), avg_val_HD[0].item(), avg_val_MSD[0].item()]
        df.loc[1] = ['1pt', avg_val_F1[1].item(), avg_val_Dice[1].item(), avg_val_HD[1].item(), avg_val_MSD[1].item()]
        df.loc[2] = ['2capsule', avg_val_F1[2].item(), avg_val_Dice[2].item(), avg_val_HD[2].item(), avg_val_MSD[2].item()]
        df.loc[3] = ['3tuft', avg_val_F1[3].item(), avg_val_Dice[3].item(), avg_val_HD[3].item(), avg_val_MSD[3].item()]
        df.loc[4] = ['4vessel', avg_val_F1[4].item(), avg_val_Dice[4].item(), avg_val_HD[4].item(), avg_val_MSD[4].item()]
        df.loc[5] = ['5ptc', avg_val_F1[5].item(), avg_val_Dice[5].item(), avg_val_HD[5].item(), avg_val_MSD[5].item()]
        df.to_csv(os.path.join(output_folder,'testing_result.csv'))

        single_df_0.to_csv(os.path.join(output_folder,'testing_result_0.csv'))
        single_df_1.to_csv(os.path.join(output_folder,'testing_result_1.csv'))
        single_df_2.to_csv(os.path.join(output_folder,'testing_result_2.csv'))
        single_df_3.to_csv(os.path.join(output_folder,'testing_result_3.csv'))
        single_df_4.to_csv(os.path.join(output_folder,'testing_result_4.csv'))
        single_df_5.to_csv(os.path.join(output_folder,'testing_result_5.csv'))

    end = timeit.default_timer()
    print(end - start, 'seconds')


if __name__ == '__main__':
    main()
