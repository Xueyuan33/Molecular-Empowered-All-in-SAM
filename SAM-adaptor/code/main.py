import argparse
import yaml
import torch
import sys
from train_test import *
import numpy as np
import random

random_seed = 2023
torch.cuda.manual_seed_all(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)
np.random.seed(random_seed)
os.environ['PYTHONHASHSEED'] = str(random_seed)
torch.cuda.manual_seed_all(random_seed)


def _init_fn(random_seed):
    np.random.seed(int(random_seed))


def seed_worker(random_seed):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


########################################################################################################################

if __name__ == '__main__':
    # time.sleep(300)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="configs/Example.yaml")  # sam-vit-b_pseudo_3_crop_selfTrain_Train
    parser.add_argument('--name', default='Testcode')
    #parser.add_argument('--tag', default=None)
    parser.add_argument("--local_rank", type=int, default=0, help="")
    parser.add_argument("--orisize", type=int, default=1000, help="")  # Original size of images
    parser.add_argument("--patience_max", type=int, default=50, help="Train until the validation loss hasn't decreased for patience_max consecutive epochs")
    parser.add_argument("--test_epoch_start", type=int, default=50, help="Do not do testing per epoch until the test_epoch_start'th epoch")
    args = parser.parse_args()

    args.random_seed = random_seed
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)

    print(args)

    ### Load config
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        if args.local_rank == 0:
            print('config loaded.')

    args.save_dir = os.path.join('./save', args.name)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    sys.stdout = open(args.save_dir + '/logfile.txt', 'a') # log file


    ### Load data
    train_loader, val_loader, test_loader = make_data_loaders(config, args)

    model, model.optimizer, model.epoch_start, model.lr_scheduler = prepare_training(config, args)

    ### Model initializatin + Train + Val
    main(config, args.save_dir, args, train_loader, val_loader, test_loader, model)

    ### Test
    test_main(args, model, args.save_dir, test_loader)


    sys.stdout.close()
