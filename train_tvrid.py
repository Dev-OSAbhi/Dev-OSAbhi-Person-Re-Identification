import os
import argparse
import torch
import torchreid
from torchreid import models, utils
from torchreid.data import ImageDataManager
from torchreid.engine import ImageSoftmaxEngine
from torchreid.optim import build_optimizer, build_lr_scheduler
from torchreid.data.datasets.image.tvrid import TVRID

def main():
    parser = argparse.ArgumentParser(description='Train TVRID model with Torchreid')
    parser.add_argument('--root', type=str, default='data/DB_extracted', help='path to dataset')
    parser.add_argument('--track', type=str, default='rgb', choices=['rgb', 'depth', 'cross'], help='competition track')
    parser.add_argument('--model', type=str, default='osnet_x1_0', help='model name')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--max-epochs', type=int, default=60, help='max epochs')
    parser.add_argument('--lr', type=float, default=0.003, help='learning rate')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--eval-freq', type=int, default=-1, help='evaluation frequency (-1 to disable)')
    parser.add_argument('--save-dir', type=str, default='log/tvrid', help='output directory')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Register dataset
    torchreid.data.register_image_dataset('tvrid', TVRID)

    # Initialize datamanager
    # We pass 'tvrid_track' to dataset_kwargs so TVRID class receives it
    datamanager = ImageDataManager(
        root=args.root,
        sources='tvrid',
        targets='tvrid',
        height=256,
        width=128,
        batch_size_train=args.batch_size,
        batch_size_test=100,
        transforms=['random_flip', 'random_crop'], # standard reid transforms
        tvrid_track=args.track 
    )

    # Build model
    model = models.build_model(
        name=args.model,
        num_classes=datamanager.num_train_pids,
        loss='softmax',
        pretrained=True
    )
    
    if torch.cuda.is_available():
        model = model.cuda()

    # Build optimizer
    optimizer = build_optimizer(
        model,
        optim='adam',
        lr=args.lr
    )

    # Build scheduler
    scheduler = build_lr_scheduler(
        optimizer,
        lr_scheduler='multi_step',
        stepsize=[30, 50],
        max_epoch=args.max_epochs
    )

    # Build engine
    engine = ImageSoftmaxEngine(
        datamanager,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        label_smooth=True
    )

    # Run training
    engine.run(
        save_dir=args.save_dir,
        max_epoch=args.max_epochs,
        eval_freq=args.eval_freq,
        print_freq=10,
        test_only=False
    )

if __name__ == '__main__':
    main()
