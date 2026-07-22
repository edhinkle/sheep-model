# Modeled after: https://github.com/NERSC/nersc-dl-multigpu/blob/main/train_multi_gpu.py

import os, sys, time
from collections import OrderedDict
import argparse
from xml.parsers.expat import model
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torch.multiprocessing as mp
import numpy as np
sys.path.insert(0, '/global/cfs/cdirs/dune/users/ehinkle/nd_prototypes_ana/sheep-model/models/cnn/train_sheep_cnn_nersc/utils/')
from utils.parse_yaml import ParseYAML
from utils.data_loader import get_data_loader
from utils.custom_loss import WeightedMSELoss, WeightedL1Loss
import yaml
import torch.optim as optim
from torch.optim import lr_scheduler
import csv
import MinkowskiEngine as ME
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
        self.log_train_preds = params.log_train_preds
        self.log_val_preds = params.log_val_preds
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
        self.params['batch_stats_log_path'] = os.path.join(exp_dir, 'logs/{}_{}_batch_stats_log.csv'.format(self.run_num, self.config))
        self.params['train_pred_log_path'] = os.path.join(exp_dir, 'logs/{}_{}_train_pred_log.csv'.format(self.run_num, self.config))
        self.params['val_pred_log_path'] = os.path.join(exp_dir, 'logs/{}_{}_val_pred_log.csv'.format(self.run_num, self.config))
        self.params['resuming'] = True if os.path.isfile(self.params.checkpoint_path) else False

    def launch(self):
        exp_dir = os.path.join(*[self.results_dir, self.config, self.run_num])
        self.init_exp_dir(exp_dir)

        # Set up logging to file
        if self.world_rank == 0 and self.params['resuming']==False:
            with open(self.params['log_path'], 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'train_iter', 'train_loss', 'val_loss', 'train_time', 'val_time', 'train_avg_active_pixels', 'val_avg_active_pixels'])

            # Set up batch norm stats logging to file
            with open(self.params['batch_stats_log_path'], 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'layer_name', 'global_batch_mean', 'global_batch_var', 'running_mean', 'running_var', 'momentum'])

            # Set up train prediction logging to file
            if self.log_train_preds == True:
                with open(self.params['train_pred_log_path'], 'w') as f:
                    writer = csv.writer(f)
                    writer.writerow(['idx', 'label', 'prediction', 'visible_energy', 've_frac', 'mg_frac', 'oob_frac', 'start_position', 'rotation_matrix'])
            
            # Set up validation prediction logging to file
            if self.log_val_preds == True:
                with open(self.params['val_pred_log_path'], 'w') as f:
                    writer = csv.writer(f)
                    writer.writerow(['idx', 'label', 'prediction', 'visible_energy', 've_frac', 'mg_frac', 'oob_frac', 'start_position', 'rotation_matrix'])

        self.params['global_batch_size'] = self.params.batch_size
        self.params['local_batch_size'] = int(self.params.batch_size//self.world_size)
        self.params['global_valid_batch_size'] = self.params.valid_batch_size
        self.params['local_valid_batch_size'] = int(self.params.valid_batch_size//self.world_size)

        # Switch to file_system strategy
        # Must be called before creating any DataLoader workers
        mp.set_sharing_strategy('file_system')

        # get the dataloaders
        self.train_data_loader, self.train_sampler = get_data_loader(self.params, self.params.train_path, dist.is_initialized(), train=True)
        #self.test_data_loader, self.test_sampler = get_data_loader(self.params, self.params.test_path, dist.is_initialized(), train=False)
        self.val_data_loader, _ = get_data_loader(self.params, self.params.val_path, dist.is_initialized(), train=False)

        # get the model
        self.model = models.sheep_cnn.sheep_cnn(self.params).to(self.device, non_blocking=True)
        # convert batch norm layers to sync batch norm for distributed training
        if dist.is_initialized():
            self.model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(self.model)
        for name, module in self.model.named_modules():
            if isinstance(module, ME.MinkowskiSyncBatchNorm):
                module.register_forward_hook(models.sheep_cnn.bn_hook(name))

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total parameters: {total_params:,}")

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
        self.schedulerConstantLR = lr_scheduler.ConstantLR(self.optimizer, factor=self.params.lr_start_factor, total_iters=self.params.lr_epochs_low)
        self.schedulerExponentialLR = lr_scheduler.ExponentialLR(self.optimizer, gamma=self.params.lr_decay_gamma)
        self.scheduler = self.schedulerConstantLR

        # set loss functions
        if self.params.loss_fn == 'MSELoss':
            self.loss_func = torch.nn.MSELoss()
        elif self.params.loss_fn == 'WeightedMSELoss':
            self.loss_func = WeightedMSELoss()
        elif self.params.loss_fn == 'L1Loss':
            self.loss_func = torch.nn.L1Loss()
        elif self.params.loss_fn == 'WeightedL1Loss':
            self.loss_func = WeightedL1Loss()
        elif self.params.loss_fn == 'HuberLoss':
            self.loss_func = torch.nn.HuberLoss()

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
                print("Setting epoch {} for train sampler and datasets...".format(epoch))
                self.train_sampler.set_epoch(epoch)
                print("Train sampler epoch set to {}".format(self.train_sampler.epoch))
                self.train_data_loader.dataset._set_epoch(epoch) # <-- added for deterministic per-sample RNGs
                print("Train dataset epoch set to {}".format(self.train_data_loader.dataset._epoch))
                self.val_data_loader.dataset._set_epoch(epoch) # <-- added for deterministic per-sample RNG
                print("Validation dataset epoch set to {}".format(self.val_data_loader.dataset._epoch))
            start = time.time()

            # training
            if self.log_train_preds == True:
                tr_time, self.labels, self.predictions, self.visible_energy, self.ve_frac, self.mg_frac, self.oob_frac, self.start_positions, self.rotation_matrices, self.idx = self.train_one_epoch()
                #print("Start positions:", self.start_positions)
                if self.train_logE == True:
                    self.labels = np.exp(self.labels)
                    self.predictions = np.exp(self.predictions)
                else:
                    self.labels = self.labels*self.params.energy_scaled
                    self.predictions = self.predictions*self.params.energy_scaled
            else:
                tr_time  = self.train_one_epoch()

            if dist.is_initialized():
                dist.barrier()  # <-- align all ranks following training on one epoch

            if self.log_train_preds == True:
                for i in range(len(self.labels)):
                    if self.world_rank == 0:
                        with open(self.params['train_pred_log_path'], 'a') as f:
                            writer = csv.writer(f)
                            writer.writerow([self.idx[i], self.labels[i], self.predictions[i], self.visible_energy[i], self.ve_frac[i], self.mg_frac[i], self.oob_frac[i], self.start_positions[i], self.rotation_matrices[i]])
            
                if dist.is_initialized():
                    dist.barrier()  # <-- align all ranks following training on one epoch


            # validation
            if self.log_val_preds == True:
                val_time, self.labels, self.predictions, self.visible_energy, self.ve_frac, self.mg_frac, self.oob_frac, self.start_positions, self.rotation_matrices, self.idx = self.val_one_epoch()
                #print("Start positions:", self.start_positions)
                if self.train_logE == True:
                    self.labels = np.exp(self.labels)
                    self.predictions = np.exp(self.predictions)
                else:
                    self.labels = self.labels*self.params.energy_scaled
                    self.predictions = self.predictions*self.params.energy_scaled
            else:
                val_time = self.val_one_epoch()

            if dist.is_initialized():
                dist.barrier()  # <-- align all ranks following validation on one epoch

            if self.log_val_preds == True:
                for i in range(len(self.labels)):
                    if self.world_rank == 0:
                        with open(self.params['val_pred_log_path'], 'a') as f:
                            writer = csv.writer(f)
                            writer.writerow([self.idx[i], self.labels[i], self.predictions[i], self.visible_energy[i], self.ve_frac[i], self.mg_frac[i], self.oob_frac[i], self.start_positions[i], self.rotation_matrices[i]])
            
                if dist.is_initialized():
                    dist.barrier()  # <-- align all ranks following training on one epoch


            start_after_val = time.time()
            # learning rate scheduler
            self.scheduler.step()
            for param_group in self.optimizer.param_groups:
                print(param_group['lr'])

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
                        writer.writerow([self.epoch, self.iters, self.logs['train_loss'], self.logs['val_loss'], tr_time, val_time, self.logs['train_avg_active_pixels'], self.logs['val_avg_active_pixels'],])
                    # Save batch norm stats to separate log file
                    with open(self.params['batch_stats_log_path'], 'a', newline='') as f:
                        writer = csv.writer(f)
                        # Assuming batch_norm_stats is a dict with layers as keys and their stats as values
                        for key in models.sheep_cnn.batch_norm_stats:
                            stats = models.sheep_cnn.batch_norm_stats[key]
                            writer.writerow([self.epoch, key, stats['global_batch_mean'], stats['global_batch_var'], stats['running_mean'], stats['running_var'], stats['momentum']])  # Log the stats for the current epoch
                        # Reset batch norm stats after logging
                    models.sheep_cnn.batch_norm_stats = {}
            # some print statements
            if self.log_to_screen:
                print('Time taken for epoch {} is {} sec; with {}/{} in tr/val'.format(self.epoch+1, time.time()-start, tr_time, val_time))
                print('Time taken after validation for epoch {} is {} sec'.format(self.epoch+1, time.time()-start_after_val))
                print('Loss = {}, Val loss = {}'.format(self.logs['train_loss'], self.logs['val_loss']))


    def train_one_epoch(self):
        tr_time = 0
        load_inputs_time = 0
        total_train_start_time = time.time()
        self.model.train()

        # buffers for logs
        logs_buff = torch.zeros((1), dtype=torch.float32, device=self.device)
        logs_buff_two = torch.zeros((1), dtype=torch.float32, device=self.device)
        self.logs['train_loss'] = logs_buff[0].view(-1)
        self.logs['train_avg_active_pixels'] = logs_buff_two[0].view(-1)
        if self.log_to_screen:
            print("Starting epoch {} with {} batches".format(self.epoch+1, len(self.train_data_loader)))

        if self.log_train_preds == True:
            labels = []
            preds = []
            visible_energy = []
            ve_frac = []
            mg_frac = []
            oob_frac = []
            start_positions = []
            rotation_matrices = []
            idxs = []

        end_of_last_step = time.time()
        for i, (inputs, targets, VE_frac, MG_frac, OOB_frac, start_pos, rot_mat, idx) in enumerate(self.train_data_loader):
            self.iters += 1
            #print("Inputs: ", inputs.size())
            inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
            #print("Active pixels:",inputs.shape[0])
            tr_start = time.time()

            #self.model.zero_grad()
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            print(f"rank {self.world_rank} batch {i} outputs.shape={outputs.shape}, targets.shape={targets.shape}")

            if self.log_train_preds == True:
                VE_frac, MG_frac, OOB_frac, start_pos, rot_mat = VE_frac.to(self.device), MG_frac.to(self.device), OOB_frac.to(self.device), start_pos.to(self.device), rot_mat.to(self.device)
                outputs = self.model(inputs)
                labels.append(targets.detach().reshape(-1))
                preds.append(outputs.detach().reshape(-1))
                ve_frac.append(VE_frac.detach().reshape(-1))
                mg_frac.append(MG_frac.detach().reshape(-1))
                oob_frac.append(OOB_frac.detach().reshape(-1))
                start_positions.append(start_pos.detach())
                rotation_matrices.append(rot_mat.detach())
                idxs.append(idx.detach())

                # Get VE 
                batch_ids = inputs[:,0].long()
                visible_energy_values = inputs[:,4]
                assert isinstance(visible_energy_values, torch.Tensor)
                num_batches = int(batch_ids.max().item()) + 1
                visible_energy_sums = torch.zeros(num_batches, device=batch_ids.device)
                visible_energy_sums = visible_energy_sums.scatter_add(0, batch_ids, visible_energy_values)
                visible_energy.append(visible_energy_sums.detach())


            loss = self.loss_func(outputs, targets)
            #if self.log_to_screen:
            #    print("Train loss batch {}: {}".format(i, loss.item()))
            loss.backward()
            self.optimizer.step()

            # Look at memory usage
            if i % 1 == 0:
                # current usage (bytes)
                a = torch.cuda.memory_allocated()
                r = torch.cuda.memory_reserved()  # "cached" by allocator
                m = torch.cuda.max_memory_allocated()
                print(f"step {i}: alloc={a/1e6:.1f} MB res={r/1e6:.1f} MB max={m/1e6:.1f} MB")
 
            # add all the minibatch losses
            #print("Training loss:", loss.detach())
            self.logs['train_loss'] += loss.detach()
            self.logs['train_avg_active_pixels'] += inputs.shape[0]
            #print("Total active pixels:", self.logs['train_avg_active_pixels'])

            tr_time += time.time() - tr_start
            load_inputs_time += tr_start - end_of_last_step
            end_of_last_step = time.time()

            #print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
            #print(f"Reserved:  {torch.cuda.memory_reserved()/1e9:.2f} GB")
        #print("Train loss before data loader adjustment:", self.logs['train_loss'])
        #print("Train active pixels before data loader adjustment:", self.logs['train_avg_active_pixels'])
        print("Input loading time for epoch {}: {} sec".format(self.epoch+1, load_inputs_time))
        self.logs['train_loss'] /= len(self.train_data_loader)
        #print("Train loss after data loader adjustment:", self.logs['train_loss'])
        self.logs['train_avg_active_pixels'] /= len(self.train_data_loader)
        print("Train Data loader length:", len(self.train_data_loader))

        # reset the peak counter to measure next epoch separately
        torch.cuda.reset_max_memory_allocated()

        logs_to_reduce = ['train_loss', 'train_avg_active_pixels']
        if dist.is_initialized(): # reduce the logs across multiple GPUs
            print("World size: {}, reducing logs across GPUs...".format(dist.get_world_size()))
            for key in logs_to_reduce:
                dist.all_reduce(self.logs[key].detach())
                self.logs[key] = float(self.logs[key]/dist.get_world_size())

        tr_time_total = time.time() - total_train_start_time
        print("Total training time for epoch {}: {} sec".format(self.epoch+1, tr_time_total))

        if self.log_train_preds == True:
            return tr_time, torch.concat(labels).cpu().numpy(), torch.concat(preds).cpu().numpy(), torch.concat(visible_energy).cpu().numpy(), torch.concat(ve_frac).cpu().numpy(), torch.concat(mg_frac).cpu().numpy(), torch.concat(oob_frac).cpu().numpy(), torch.concat(start_positions).cpu().numpy(), torch.concat(rotation_matrices).cpu().numpy(), torch.concat(idxs).cpu().numpy()
        else:
            return tr_time

    def val_one_epoch(self):
        self.model.eval()
        val_start = time.time()

        logs_buff = torch.zeros((1), dtype=torch.float32, device=self.device)
        logs_buff_two = torch.zeros((1), dtype=torch.float32, device=self.device)
        self.logs['val_loss'] = logs_buff[0].view(-1)
        self.logs['val_avg_active_pixels'] = logs_buff_two[0].view(-1)
        if self.log_to_screen:
            print("Starting validation with {} batches".format(len(self.val_data_loader)))

        if self.log_val_preds == True:
            labels = []
            preds = []
            visible_energy = []
            ve_frac = []
            mg_frac = []
            oob_frac = []
            start_positions = []
            rotation_matrices = []
            idxs = []

        with torch.no_grad():
            for i, (inputs, targets, VE_frac, MG_frac, OOB_frac, start_pos, rot_mat, idx) in enumerate(self.val_data_loader):
                inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
                outputs = self.model(inputs)
                loss = self.loss_func(outputs, targets)

                #if self.log_to_screen:
                #    print("Val loss batch {}: {}".format(i, loss.item()))
                self.logs['val_loss'] += loss.detach()
                self.logs['val_avg_active_pixels'] += inputs.shape[0] 


                if self.log_val_preds == True:
                    VE_frac, MG_frac, OOB_frac, start_pos, rot_mat = VE_frac.to(self.device), MG_frac.to(self.device), OOB_frac.to(self.device), start_pos.to(self.device), rot_mat.to(self.device)
                    outputs = self.model(inputs)
                    labels.append(targets.detach().reshape(-1))
                    preds.append(outputs.detach().reshape(-1))
                    ve_frac.append(VE_frac.detach().reshape(-1))
                    mg_frac.append(MG_frac.detach().reshape(-1))
                    oob_frac.append(OOB_frac.detach().reshape(-1))
                    start_positions.append(start_pos.detach())
                    rotation_matrices.append(rot_mat.detach())
                    idxs.append(idx.detach())

                    # Get VE 
                    batch_ids = inputs[:,0].long()
                    visible_energy_values = inputs[:,4]
                    assert isinstance(visible_energy_values, torch.Tensor)
                    num_batches = int(batch_ids.max().item()) + 1
                    visible_energy_sums = torch.zeros(num_batches, device=batch_ids.device)
                    visible_energy_sums = visible_energy_sums.scatter_add(0, batch_ids, visible_energy_values)
                    visible_energy.append(visible_energy_sums.detach())


        self.logs['val_loss'] /= len(self.val_data_loader)
        self.logs['val_avg_active_pixels'] /= len(self.val_data_loader)
        print("Length of validation data loader:", len(self.val_data_loader))
        if dist.is_initialized():
            for key in ['val_loss', 'val_avg_active_pixels']:
                dist.all_reduce(self.logs[key].detach())
                self.logs[key] = float(self.logs[key]/dist.get_world_size())

        val_time = time.time() - val_start

        if self.log_val_preds == True:
            return val_time, torch.concat(labels).cpu().numpy(), torch.concat(preds).cpu().numpy(), torch.concat(visible_energy).cpu().numpy(), torch.concat(ve_frac).cpu().numpy(), torch.concat(mg_frac).cpu().numpy(), torch.concat(oob_frac).cpu().numpy(), torch.concat(start_positions).cpu().numpy(), torch.concat(rotation_matrices).cpu().numpy(), torch.concat(idxs).cpu().numpy()
        else:
            return val_time

    def save_checkpoint(self, checkpoint_path, is_best=False, model=None):
        if not model:
            model = self.model

        # Save persistent checkpoints for every tenth epoch
        if self.epoch % 10 == 0 and self.epoch > 0:
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

        total = 0
        for p in self.model.parameters():
            total += p.abs().sum().item()
        print("Checkpoint loaded: epoch {}, iters {}, total abs param sum {}".format(self.startEpoch, self.iters, total))

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