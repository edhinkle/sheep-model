# Modeled after: https://github.com/NERSC/nersc-dl-multigpu/blob/main/train_multi_gpu.py

import os, sys, time
from collections import OrderedDict
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import numpy as np
sys.path.insert(0, '/global/cfs/cdirs/dune/users/ehinkle/nd_prototypes_ana/sheep-model/models/cnn/train_sheep_cnn_nersc/utils/')
from utils.parse_yaml import ParseYAML
from utils.data_loader import get_data_loader
import yaml
import torch.optim as optim
from torch.optim import lr_scheduler
import csv
import datetime

# models
import models.sheep_cnn


class Trainer():
    """ trainer class """
    def __init__(self, params, args):
        ''' init vars for distributed training (ddp) and logging'''
        self.results_dir = args.results_dir
        self.config = args.config 
        self.run_num = args.run_num
        self.world_size = 1
        if 'WORLD_SIZE' in os.environ:
            self.world_size = int(os.environ['WORLD_SIZE'])

        # Get local rank first (even if not DDP)
        if 'LOCAL_RANK' in os.environ:
            self.local_rank = int(os.environ["LOCAL_RANK"])
        else: 
            self.local_rank = 0

        # Select GPU for rank (if mulitgpu)
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)

        # Initialize DDP using NCCL backend
        if self.world_size > 1: # multigpu, use DDP with standard NCCL backend for communication routines
            dist.init_process_group(backend='nccl',
                                    init_method='env://',
                                    timeout=datetime.timedelta(minutes=60))
            self.world_rank = dist.get_rank()
        else: 
            self.world_rank =0        

        # Set cuDNN settings after selecting device
        torch.backends.cudnn.benchmark = True
        
        self.log_to_screen = (self.world_rank==0)
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.device = torch.device('cpu')
        self.params = params
        print("running on rank {} with world size {}".format(self.world_rank, self.world_size))


    def init_exp_dir(self, exp_dir):
        if self.world_rank==0:
            if not os.path.isdir(exp_dir):
                os.makedirs(exp_dir)
                os.makedirs(os.path.join(exp_dir, 'checkpoints/'))
                os.makedirs(os.path.join(exp_dir, 'logs/'))
        self.params['experiment_dir'] = os.path.abspath(exp_dir)
        self.params['checkpoint_path'] = os.path.join(exp_dir, 'checkpoints/ckpt.tar')
        self.params['log_path'] = os.path.join(exp_dir, 'logs/{}_{}_train_log.csv'.format(self.run_num, self.config))
        self.params['resuming'] = True if os.path.isfile(self.params.checkpoint_path) else False

    def launch(self):
        exp_dir = os.path.join(*[self.results_dir, self.config, self.run_num])
        self.init_exp_dir(exp_dir)

        # Set up logging to file
        if self.world_rank == 0 and self.params['resuming']==False:
            with open(self.params['log_path'], 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'train_iter', 'train_loss', 'val_loss', 'train_time', 'val_time'])

        self.params['global_batch_size'] = self.params.batch_size
        self.params['local_batch_size'] = int(self.params.batch_size//self.world_size)
        self.params['global_valid_batch_size'] = self.params.valid_batch_size
        self.params['local_valid_batch_size'] = int(self.params.valid_batch_size//self.world_size)

        # get the dataloaders
        self.train_data_loader, self.train_sampler = get_data_loader(self.params, self.params.train_path, dist.is_initialized(), train=True)
        #self.test_data_loader, self.test_sampler = get_data_loader(self.params, self.params.test_path, dist.is_initialized(), train=False)
        self.val_data_loader, _ = get_data_loader(self.params, self.params.val_path, dist.is_initialized(), train=False)

        # get the model
        self.model = models.sheep_cnn.sheep_cnn(self.params).to(self.device)

        # distributed wrapper for data parallel
        if dist.is_initialized():
            # Get model ready for distributed training
            self.model = DistributedDataParallel(self.model,
                                                device_ids=[self.local_rank],
                                                output_device=[self.local_rank])
            
            # Check that each rank has the same number of batches
            local_len = torch.tensor([len(self.train_data_loader)], device=self.device)
            world_lens = [torch.zeros_like(local_len) for _ in range(dist.get_world_size())]
            dist.all_gather(world_lens, local_len)
            if self.world_rank == 0 and self.log_to_screen:
                print("Train loader lengths per rank:", [int(t.item()) for t in world_lens])


        # set an optimizer and learning rate scheduler
        optimizer_fn = getattr(optim, self.params.optimizer)
        self.optimizer = optimizer_fn(self.model.parameters(), lr=self.params.lr)
        self.scheduler = None #lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.params.max_cosine_lr_epochs)

        # set loss functions
        if params.loss_fn == 'MSELoss':
            self.loss_func = torch.nn.MSELoss()

        # checkpointing
        self.iters = 0
        self.startEpoch = 0
        if self.params.resuming:
            print("Loading checkpoint %s"%self.params.checkpoint_path)
            self.restore_checkpoint(self.params.checkpoint_path)
        self.epoch = self.startEpoch
        self.logs = {}

        # launch training
        self.train()

    def train(self):
        if self.log_to_screen:
            print("Starting training loop...")
     
        best_loss = np.inf
        best_epoch = 0
        self.logs['best_epoch'] = best_epoch

        for epoch in range(self.startEpoch, self.params.max_epochs):
            self.epoch = epoch
            if dist.is_initialized():
                # shuffles data before every epoch
                self.train_sampler.set_epoch(epoch)
                #self.valid_sampler.set_epoch(epoch) # <-- Got rid of valid sampler
            start = time.time()

            # training
            tr_time  = self.train_one_epoch()

            if dist.is_initialized():
                dist.barrier()  # <-- align all ranks following training on one epoch

            # validation
            val_time = self.val_one_epoch_from_multi_pose_cache()

            if dist.is_initialized():
                dist.barrier()  # <-- align all ranks following validation on one epoch

            # learning rate scheduler
            #self.scheduler.step()

            # keep track of best model according to validation loss
            if self.logs['val_loss'] <= best_loss:
                is_best_loss = True
                best_loss = self.logs['val_loss']
            else:
                is_best_loss = False
            self.logs['best_val_loss'] = best_loss
            best_epoch = self.epoch if is_best_loss else best_epoch
            self.logs['best_epoch'] = best_epoch
            
            # save checkpoint (if best epoch additionally save the best epoch too)
            if self.params.save_checkpoint:
                if self.world_rank == 0:
                    #checkpoint at the end of every epoch
                    self.save_checkpoint(self.params.checkpoint_path, is_best=is_best_loss)
                    with open(self.params['log_path'], 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([self.epoch, self.iters, self.logs['train_loss'], self.logs['val_loss'], tr_time, val_time])

            # some print statements
            if self.log_to_screen:
                print('Time taken for epoch {} is {} sec; with {}/{} in tr/val'.format(self.epoch+1, time.time()-start, tr_time, val_time))
                print('Loss = {}, Val loss = {}'.format(self.logs['train_loss'], self.logs['val_loss']))


    def train_one_epoch(self):
        tr_time = 0
        self.model.train()

        # buffers for logs
        logs_buff = torch.zeros((1), dtype=torch.float32, device=self.device)
        self.logs['train_loss'] = logs_buff[0].view(-1)
        if self.log_to_screen:
            print("Starting epoch {} with {} batches".format(self.epoch+1, len(self.train_data_loader)))

        for i, (inputs, targets, VE_frac, MG_frac, OOB_frac) in enumerate(self.train_data_loader):
            self.iters += 1
            data_start = time.time()
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            print("Inputs Shape:", inputs.shape)
            print("Targets Shape:", targets.shape)
            print("Target:", targets)
            tr_start = time.time()

            #self.model.zero_grad()
            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            loss = self.loss_func(outputs, targets)
            #if self.log_to_screen:
            #    print("Train loss batch {}: {}".format(i, loss.item()))
            loss.backward()
            self.optimizer.step()
 
            # add all the minibatch losses
            print("Training loss:", loss.detach())
            self.logs['train_loss'] += loss.detach()  / len(self.train_data_loader)

            tr_time += time.time() - tr_start

        #self.logs['train_loss'] /= len(self.train_data_loader)

        logs_to_reduce = ['train_loss']
        if dist.is_initialized(): # reduce the logs across multiple GPUs
            for key in logs_to_reduce:
                dist.all_reduce(self.logs[key].detach())
                self.logs[key] = float(self.logs[key]/dist.get_world_size())

        return tr_time

    def val_one_epoch(self):
        self.model.eval()
        val_start = time.time()

        logs_buff = torch.zeros((1), dtype=torch.float32, device=self.device)
        self.logs['val_loss'] = logs_buff[0].view(-1)
        if self.log_to_screen:
            print("Starting validation with {} batches".format(len(self.val_data_loader)))

        with torch.no_grad():
            for i, (inputs, targets, VE_frac, MG_frac, OOB_frac) in enumerate(self.val_data_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_func(outputs, targets)
                #if self.log_to_screen:
                #    print("Val loss batch {}: {}".format(i, loss.item()))
                self.logs['val_loss'] += loss.detach()

        self.logs['val_loss'] /= len(self.val_data_loader)
        if dist.is_initialized():
            for key in ['val_loss']:
                dist.all_reduce(self.logs[key].detach())
                self.logs[key] = float(self.logs[key]/dist.get_world_size())

        val_time = time.time() - val_start

        return val_time
    
    def val_one_epoch_from_multi_pose_cache(self):

        val_start = time.time()
        if self.world_rank == 0: 
            self.model.eval()
            total_loss = 0.0
            total_events = 0

            logs_buff = torch.zeros((1), dtype=torch.float32, device=self.device)
            self.logs['val_loss'] = logs_buff[0].view(-1)
            if self.log_to_screen:
                print("Starting validation from multi-pose cache with {} batches".format(len(self.val_data_loader)))

            with torch.no_grad():
                for (poses, label, VE_frac, MG_frac, OOB_frac) in self.val_data_loader:
                    # Set batch size to 1:
                    poses = poses[0] # k tensors [N, K]
                    label = label[0].to(self.device) # tensor [N]
                    if self.log_to_screen:
                        print("Processing event with label shape {}".format(label.shape))
                        print("Label:", label)
                        print("Label squeezed:", label.squeeze())

                    pose_predictions = []
                    for combined_data in poses:
                        inputs = combined_data.to(self.device)
                        if self.log_to_screen:
                            print("  Input shape for pose:", inputs.shape)
                        outputs = self.model.module(inputs) # remove DDP wrapper for validation
                        #print("Shape of outputs:", outputs.shape)
                        #loss = self.loss_func(outputs, label)
                        #print("  Pose loss: {}".format(loss.detach().item()))
                        #print("  Pose loss no item{}".format(loss.detach()))
                        pose_predictions.append(outputs)

                    average_pose_prediction = torch.mean(torch.stack(pose_predictions, dim=0))
                    print("Average pose prediction shape:", average_pose_prediction.shape)
                    loss = self.loss_func(average_pose_prediction, label.squeeze())

                    total_loss += float(loss.detach().item())
                    total_events += 1

                    # Check is nan/inf:
                    if torch.isnan(average_pose_prediction) or torch.isinf(average_pose_prediction):
                        print(f"Rank {self.world_rank}: Invalid prediction: {average_pose_prediction}")
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"Rank {self.world_rank}: Invalid loss: {loss}")

                    # Check average pose predictions
                    print(f"Pred: {average_pose_prediction.item():.6f}, Label: {label.item():.6f}")

            self.logs['val_loss'] = total_loss / max(total_events, 1)
        else:
            self.logs['val_loss'] = 0.0

        if dist.is_initialized():
            val_loss_fl = float(self.logs['val_loss'])
            val_loss_t = torch.tensor(val_loss_fl, dtype=torch.float32, device=self.device)
            #dist.broadcast(val_loss_t, src=0)
            self.logs['val_loss'] = float((val_loss_t.item()))

        val_time = time.time() - val_start

        return val_time

    def save_checkpoint(self, checkpoint_path, is_best=False, model=None):
        if not model:
            model = self.model

        # Save persistent checkpoints for every fifth epoch
        if self.epoch % 5 == 0 and self.epoch > 0:
            torch.save({'iters': self.iters, 'epoch': self.epoch, 'model_state': model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(), 'scheduler_state_dict': (self.scheduler.state_dict() if self.scheduler is not None else None)}, checkpoint_path.replace('.tar', '_epoch_{}_iters{}.tar'.format(self.epoch, self.iters)))
        
        # Always save the latest checkpoint (overwritten every epoch)
        torch.save({'iters': self.iters, 'epoch': self.epoch, 'model_state': model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(), 'scheduler_state_dict': (self.scheduler.state_dict() if self.scheduler is not None else None)}, checkpoint_path)
        
        # If best epoch, save a copy of the best checkpoint
        if is_best:
            torch.save({'iters': self.iters, 'epoch': self.epoch, 'model_state': model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(), 'scheduler_state_dict': (self.scheduler.state_dict() if  self.scheduler is not None else None)}, checkpoint_path.replace('.tar', '_epoch_{}_iters_{}_best.tar'.format(self.epoch, self.iters)))

    def restore_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cuda:{}'.format(self.local_rank)) 
        try:
            self.model.load_state_dict(checkpoint['model_state'])
        except:
            new_state_dict = OrderedDict()
            for key, val in checkpoint['model_state'].items():
                name = key[7:]
                new_state_dict[name] = val 
            self.model.load_state_dict(new_state_dict)

        self.iters = checkpoint['iters']
        self.startEpoch = checkpoint['epoch'] + 1
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

if __name__ == '__main__':
    # parsers for any cmd line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config", default='./configs/default_sheep.yaml', type=str)
    parser.add_argument("--config", default='default', type=str)
    parser.add_argument("--results_dir", default='./outputs', type=str, help='directory to store results')
    parser.add_argument("--run_num", default='0', type=str, help='sub run config')
    args = parser.parse_args()
    params = ParseYAML(os.path.abspath(args.yaml_config), args.config)

    trainer = Trainer(params, args)
    trainer.launch()
    if dist.is_initialized():
        dist.barrier()

    print('Training complete')
    dist.destroy_process_group()