import yaml
from tqdm import tqdm
import torch
import torchvision
import torchvision.transforms as transforms
from statistics import mean
import matplotlib.pyplot as plt
import utils
from utils import *
import os
import numpy as np
import glob
from PIL import Image
import models
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from utils import _dice_val, _iou_val
import pandas as pd


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, mask_dir, label=0, transform=None, appendix='.png', args=None, split='test'):
        self.data_dir = data_dir
        self.mask_dir = mask_dir
        self.data_files = glob.glob(data_dir + '/*' + appendix)  # change back to tif when needed
        # self.data_files = os.listdir(data_dir)
        # self.transform = transform
        self.appendix = appendix
        self.args = args
        self.split =split #'train'
        # self.label = label
        self.transform_color = transforms.Compose([transforms.Resize(512),
                                                   # transforms.CenterCrop(1024),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                        std=[0.229, 0.224, 0.225])])

        self.transform_gray = transforms.Compose([
            transforms.Resize(512, interpolation=torchvision.transforms.InterpolationMode.NEAREST),
            # transforms.CenterCrop(1024),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),  # Normalize
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.data_files[idx])
        img = Image.open(img_path).convert('RGB')
        img_path_name = img_path.split('/')[-1].replace(self.appendix, '.png')

        mask_path = os.path.join(self.mask_dir, img_path_name)  # self.data_files[idx])
        mask = Image.open(mask_path).convert('L')
        img = self.transform_color(img).unsqueeze(0).cuda()
        mask = self.transform_gray(mask).unsqueeze(0).cuda()

        return {'inp': img, 'gt': mask, 'name': self.data_files[idx]}  # np.array(img), np.array(mask)


def make_data_loaders(config, args=None):
    """
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    test_loader = make_data_loader(config.get('test_dataset'), tag='test')

    #spec['dataset']['args']['root_path_1']
    """

    g = torch.Generator()
    g.manual_seed(args.random_seed)

    # transform = transform_color
    spec = config.get('train_dataset')
    img_dir = spec['dataset']['args']['root_path_1']  # '/home/VANDERBILT/cuic4/Research/CellSeg/Data_splits_new/train/'
    mask_dir = spec['dataset']['args']['root_path_2']  # '/home/VANDERBILT/cuic4/Research/CellSeg/Data/MoNuSeg_mask/'
    trainset = MyDataset(img_dir, mask_dir, args=args, split='train')
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=0,
                                               drop_last=False, generator=g)

    spec = config.get('val_dataset')
    img_dir = spec['dataset']['args']['root_path_1']  # '/home/VANDERBILT/cuic4/Research/CellSeg/Data_splits_new/train/'
    mask_dir = spec['dataset']['args']['root_path_2']  # '/home/VANDERBILT/cuic4/Research/CellSeg/Data/MoNuSeg_mask/'
    testset = MyDataset(img_dir, mask_dir, appendix='.png', args=args)
    val_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=0,
                                             drop_last=False, generator=g)

    spec = config.get('test_dataset')
    img_dir = spec['dataset']['args']['root_path_1']  # '/home/VANDERBILT/cuic4/Research/CellSeg/Data_splits_new/train/'
    mask_dir = spec['dataset']['args']['root_path_2']  # '/home/VANDERBILT/cuic4/Research/CellSeg/Data/MoNuSeg_mask/'
    testset = MyDataset(img_dir, mask_dir, appendix='.png', args=args)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=0,
                                              drop_last=False, generator=g)

    return train_loader, val_loader, test_loader

def prepare_training(config,args):
    if config.get('resume') is not None:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = config.get('resume') + 1
    else:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 1
    max_epoch = config.get('epoch_max')
    lr_scheduler = CosineAnnealingLR(optimizer, max_epoch, eta_min=config.get('lr_min'))
    if args.local_rank == 0:
        log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    return model, optimizer, epoch_start, lr_scheduler

def train(train_loader, model, args):
    model.train()

    if args.local_rank == 0:
        pbar = tqdm(total=len(train_loader), leave=False, desc='train')
    else:
        pbar = None

    loss_list = []
    for batch in train_loader:
        for k, v in batch.items():
            try:
                batch[k] = v.to(args.device)
            except:
                batch[k] = v[0]

        inp = batch['inp']
        gt = batch['gt']

        model.set_input(inp, gt)
        model.optimize_parameters()
        batch_loss = model.loss_G
        loss_list.append(batch_loss)
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    loss = [i.item() for i in loss_list]
    return mean(loss)


def test_new(train_loader, model, save_dir, split='', args=None, save_results=False):
    model.eval()
    save_fig_dir = save_dir + '/' + split + '/'
    if not os.path.exists(save_fig_dir):
        os.mkdir(save_fig_dir)

    f1_list, dice_list, precision_list, recall_list, best_f1_list, auc_list, iou_list, aji_list = [], [], [], [], [], [], [], []

    with torch.no_grad():
        for i_, batch in enumerate(train_loader):
            for k, v in batch.items():
                try:
                    batch[k] = v.to(args.device)
                except:
                    batch[k] = v[0]

            threshold = 0.5
            inp = batch['inp']
            pred_bin = torch.sigmoid(model.infer(inp))
            name_ = batch['name']
            transform_temp = transforms.Resize(args.orisize)
            save_output = transform_temp(pred_bin)
            save_output = save_output[0, 0, :, :] * 255
            save_output = np.array(save_output.detach().cpu())
            print('pred_bin:', pred_bin)
            print('batch['gt']', batch['gt'])







            # Save predictions
            cv2.imwrite(args.save_dir + '/' + split + '/' + name_.split('/')[-1].replace('tif', '.png'), save_output)

            batch_bestf1, batch_auc, _, _ = calc_f1(pred_bin, batch['gt'])
            pred_bin = torch.where(pred_bin >= threshold, torch.ones_like(pred_bin), torch.zeros_like(pred_bin))
            batch_dice = _dice_val(pred_bin, batch['gt'])
            batch_f1, batch_precision, batch_recall, batch_aji = calc_binf1(pred_bin, batch['gt'])
            batch_iou = _iou_val(pred_bin, batch['gt'])


            # Note that dice and f1, iou and aji are expected to have same values
            dice_list.append(batch_dice)  # .detach().cpu())
            f1_list.append(batch_f1)  # .detach().cpu())
            auc_list.append(batch_auc)
            recall_list.append(batch_recall)
            precision_list.append(batch_precision)
            best_f1_list.append(batch_bestf1)
            iou_list.append(batch_iou)
            aji_list.append(batch_aji)

            print(name_)
            print(batch_dice)

    dice_list = [i.item() for i in dice_list]
    f1_list = [i.item() for i in f1_list]
    auc_list = [i.item() for i in auc_list]
    recall_list = [i.item() for i in recall_list]
    precision_list = [i.item() for i in precision_list]
    best_f1_list = [i.item() for i in best_f1_list]
    iou_list = [i.item() for i in iou_list]
    aji_list = [i.item() for i in aji_list]
    # aji_list
    print('dice_list:', dice_list)
    print('auc_list:', auc_list)
    print('recall_list:', recall_list)
    print('precision_list:', precision_list)
    print('best_f1_list:', best_f1_list)
    print('iou_list:', iou_list)
    print('aji_list:', aji_list)

    if save_results:
        df = pd.DataFrame([[ '{0:.4f}'.format(mean(dice_list)), '{0:.4f}'.format(mean(f1_list)),  '{0:.4f}'.format(mean(auc_list)), '{0:.4f}'.format(mean(recall_list)), '{0:.4f}'.format(mean(precision_list)),'{0:.4f}'.format(mean(best_f1_list)), '{0:.4f}'.format(mean(iou_list)), '{0:.4f}'.format(mean(aji_list)) ]], columns=['dice', 'f1', 'auc', 'recall', 'precision', 'bestf1', 'iou', 'aji'])
        df.insert(0, 'name',  args.name)
        df.to_csv(args.save_dir + '/evaluation.csv', header=None, sep='\t', mode='a')

    return mean(dice_list), mean(f1_list), mean(auc_list), mean(recall_list), mean(precision_list), mean(
        best_f1_list), mean(iou_list), mean(aji_list)


def val_new(train_loader, model, args):
    model.eval()

    dice_list = []
    loss_list = []

    with torch.no_grad():
        for i_, batch in enumerate(train_loader):
            for k, v in batch.items():
                try:
                    batch[k] = v.to(args.device)
                except:
                    batch[k] = v[0]
            inp = batch['inp']
            pred = torch.sigmoid(model.infer(inp))
            batch_loss = model.loss_G_cathyAdd(pred, batch['gt'])
            threshold = 0.5
            pred_bin = torch.where(pred >= threshold, torch.ones_like(pred), torch.zeros_like(pred))

            batch_dice = _dice_val(pred_bin, batch['gt'])
            loss_list.append(batch_loss)
            dice_list.append(batch_dice)

    loss_list = [i.item() for i in loss_list]
    dice_list = [i.item() for i in dice_list]
    return mean(loss_list), mean(dice_list)


def main(config, save_path, args, train_loader, val_loader, test_loader, model):
    #global log, writer, log_info
    log, writer = utils.set_save_path(save_path, remove=False)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    #log_file = open(save_path + '/logfile.txt', 'a')
    #sys.stdout = log_file

    patience_cnt = 0

    model = model.cuda()

    # load pretrained weights from SAM
    sam_checkpoint = torch.load(config['sam_checkpoint'])
    model.load_state_dict(sam_checkpoint, strict=False)

    ## Freeze the encoder, except the prompt_generator
    for name, para in model.named_parameters():
        if "image_encoder" in name and "prompt_generator" not in name:
            para.requires_grad_(False)
    if args.local_rank == 0:
        model_total_params = sum(p.numel() for p in model.parameters())
        model_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('model_grad_params:' + str(model_grad_params), '\nmodel_total_params:' + str(model_total_params))

    ## Training Initialization
    epoch_max = config['epoch_max']
    max_val_v = -1e18 if config['eval_type'] != 'ber' else 1e8
    min_val_v = 1e18
    #timer = utils.Timer()
    best_epoch = 0
    plot_training_curve_list = {'train_loss_avg': [], 'train_dice_avg': [], 'val_dice_avg': [], 'val_loss_avg': []} # For ploting

    ## Training
    for epoch in range(model.epoch_start, epoch_max + 1):
        #t_epoch_start = timer.t()
        train_loss_G = train(train_loader, model, args=args)
        model.lr_scheduler.step()

        if args.local_rank == 0:
            log_info = ['epoch {}/{}'.format(epoch, epoch_max)]
            writer.add_scalar('lr', model.optimizer.param_groups[0]['lr'], epoch)
            log_info.append('train G: loss={:.4f}'.format(train_loss_G))
            writer.add_scalars('loss', {'train G': train_loss_G}, epoch)

            model_spec = config['model']
            model_spec['sd'] = model.state_dict()
            optimizer_spec = config['optimizer']
            optimizer_spec['sd'] = model.optimizer.state_dict()

            # Save the last model
            save(config, model, save_path, 'last')

        # Validation per epoch. Validate the training set for plotting and evaluation. Can be deleted for efficiency.
        train_loss_avg, train_dice_avg = val_new(train_loader, model, args)

        plot_training_curve_list['train_loss_avg'].append(train_loss_avg)
        plot_training_curve_list['train_dice_avg'].append(train_dice_avg)

        # Validation per epoch.
        val_loss_avg, val_dice_avg = val_new(val_loader, model, args)

        plot_training_curve_list['val_dice_avg'].append(val_dice_avg)
        plot_training_curve_list['val_loss_avg'].append(val_loss_avg)

        print('[Epoch]', epoch, ' Train_Loss', '{0:.4f}'.format(train_loss_G), 'Avg_train_Dice:',
              '{0:.4f}'.format(train_dice_avg), ' Train_Loss', '{0:.4f}'.format(train_loss_avg), 'Avg_val_Dice:',
              '{0:.4f}'.format(val_dice_avg), 'Avg_val_Loss:', '{0:.4f}'.format(val_loss_avg))

        # Best epoch based on loss
        if val_loss_avg < min_val_v:
            min_val_v = val_loss_avg
            save(config, model, save_path, 'best')
            patience_cnt = 0
            best_epoch = epoch

            test_dice_avg = None
            test_loss_avg = None

            # Validation per epoch after test_epoch_start. Validate the testing set for plotting and evaluation. Can be deleted for efficiency.
            if epoch >= args.test_epoch_start:
                test_loss_avg, test_dice_avg = val_new(test_loader, model, args)

            print('*** Best Epoch: ', best_epoch, ' Train loss: ', np.array(train_loss_G), ' Val Dice: ',
                  np.array(val_dice_avg), ' Test Dice: ',
                  np.array(test_dice_avg))
        else:
            patience_cnt = patience_cnt + 1 # No consecutive better model

        # Train until the validation loss hasn't decreased for patience_max consecutive epochs.
        if patience_cnt == args.patience_max: # Early break
            break

    plt.plot(plot_training_curve_list['train_loss_avg'], 'o-', color='r')
    plt.plot(plot_training_curve_list['val_loss_avg'], 'o-', color='g')
    plt.title('train_loss_avg')
    plt.savefig(save_path + '/train_loss.png')
    plt.close()
    plt.plot(plot_training_curve_list['train_dice_avg'], 'o-', color='r')
    plt.plot(plot_training_curve_list['val_dice_avg'], 'o-', color='g')
    plt.title('train_dice_avg')
    plt.savefig(save_path + '/train_dice.png')
    plt.close()

    return model

def test_main(args, model, save_path, test_loader):
    print('********************Testing****************************')
    args.model = os.path.join(save_path, f"model_epoch_best.pth")
    sam_checkpoint = torch.load(args.model, map_location='cuda:0')
    model.load_state_dict(sam_checkpoint, strict=True)
    test_dice_avg, test_f1_avg, test_auc_avg, test_recall_avg, test_precision_avg, test_bestf1_avg, test_iou_avg, test_aji_avg = test_new(
        test_loader, model, args.save_dir, split='test',
        args=args,save_results=True)
    print('Avg_testing_Dice:', test_dice_avg, ' Avg_testing_F1:', test_f1_avg, ' Avg_testing_AUC:', test_auc_avg,
          ' Avg_testing_Recall:', test_recall_avg, ' Avg_testing_Precision:', test_precision_avg,
          ' Avg_testing_bestf1:', test_bestf1_avg, ' Avg_testing_IoU:', test_iou_avg, 'test_aji_avg:', test_aji_avg)
    print('check****************Testing***************************')
    args.model = os.path.join(save_path, f"model_epoch_last.pth")
    sam_checkpoint = torch.load(args.model, map_location='cuda:0')
    model.load_state_dict(sam_checkpoint, strict=True)
    #test_loss_avg, test_dice_avg = val_new(val_loader, model)
    #print('Avg_val_Dice:', test_dice_avg)



def save(config, model, save_path, name):
    if config['model']['name'] == 'segformer' or config['model']['name'] == 'setr':
        if config['model']['args']['encoder_mode']['name'] == 'evp':
            prompt_generator = model.encoder.backbone.prompt_generator.state_dict()
            decode_head = model.encoder.decode_head.state_dict()
            torch.save({"prompt": prompt_generator, "decode_head": decode_head},
                       os.path.join(save_path, f"prompt_epoch_{name}.pth"))
        else:
            torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{name}.pth"))
    else:
        torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{name}.pth"))
