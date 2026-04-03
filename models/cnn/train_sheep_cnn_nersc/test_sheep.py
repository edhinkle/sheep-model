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
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import MinkowskiEngine as ME
import csv
import datetime

# models
import models.sheep_cnn

class Tester():
    """ trainer class """

    def __init__(self, params, args):

        self.results_dir = args.results_dir
        self.config = args.config 
        self.train_config = args.train_config
        self.run_num = args.run_num
        self.checkpoint_file = args.checkpoint_file
        self.params = params
        self.log_to_screen = 1 # print to screen
        self.train_logE = params.train_logE

        self.world_size = 1
        if 'WORLD_SIZE' in os.environ:
            self.world_size = int(os.environ['WORLD_SIZE'])

        # Get local rank first (even if not DDP)
        if 'LOCAL_RANK' in os.environ:
            self.local_rank = int(os.environ["LOCAL_RANK"])
        else: 
            self.local_rank = 0


        # Initialize DDP using NCCL backend
        print("Initializing DDP with world size {} and local rank {}".format(self.world_size, self.local_rank))
        if self.world_size > 1: # multicpu, use DDP with standard NCCL backend for communication routines
            dist.init_process_group(backend='gloo',
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
            self.device = torch.device("cpu")
        print("running on rank {} with world size {}".format(self.world_rank, self.world_size))


    def init_exp_dir(self, exp_dir, train_dir):
        if self.world_rank==0:
            if not os.path.isdir(exp_dir):
                os.makedirs(exp_dir)
                os.makedirs(os.path.join(exp_dir, 'logs/'))
                os.makedirs(os.path.join(exp_dir, 'plots/'))
            if not os.path.isdir(train_dir):
                raise ValueError(f"Training directory {train_dir} does not exist. Please train the model before testing.")
        self.params['experiment_dir'] = os.path.abspath(exp_dir)
        self.params['train_dir'] = os.path.abspath(train_dir)
        self.params['log_path'] = os.path.join(exp_dir, 'logs/{}_{}_{}_log.csv'.format(self.run_num, self.config, self.checkpoint_file.split('.')[0]))
        self.params['checkpoint_path'] = os.path.join(train_dir, 'checkpoints/'+self.checkpoint_file)
        self.params['resuming'] = True if os.path.isfile(self.params.checkpoint_path) else False

    def launch(self):
        exp_dir = os.path.join(*[self.results_dir, self.config, self.run_num])
        train_dir = os.path.join(*[self.results_dir, self.train_config, self.run_num])
        self.init_exp_dir(exp_dir, train_dir)

        if self.world_rank == 0:
            with open(self.params['log_path'], 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['label', 'prediction', 'visible_energy', 've_frac', 'mg_frac', 'oob_frac', 'start_position', 'rotation_matrix'])


        self.params['global_batch_size'] = self.params.batch_size
        self.params['local_batch_size'] = self.params.batch_size
        self.params['global_valid_batch_size'] = self.params.valid_batch_size
        self.params['local_valid_batch_size'] = self.params.valid_batch_size
        self.params['global_test_batch_size'] = self.params.test_batch_size
        self.params['local_test_batch_size'] = self.params.test_batch_size

        # get the dataloaders
        self.test_data_loader, self.test_sampler = get_data_loader(self.params, self.params.test_path, distributed=False, train=False, test=True)

        # get the model
        self.model = models.sheep_cnn.sheep_cnn(self.params).to(self.device)
        # convert batch norm layers to sync batch norm for distributed training
        #if dist.is_initialized():
        self.model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(self.model)
        #for name, module in self.model.named_modules():
        #    if isinstance(module, ME.MinkowskiSyncBatchNorm):
        #        module.register_forward_hook(models.sheep_cnn.bn_hook(name))

                        # distributed wrapper for data parallel
        if dist.is_initialized():
            print("Wrapping model in DistributedDataParallel on rank {}".format(self.world_rank))
            # Get model ready for distributed training
            self.model = DistributedDataParallel(self.model)
            
            # Check that each rank has the same number of batches
            local_len = torch.tensor([len(self.test_data_loader)], device=self.device)
            world_lens = [torch.zeros_like(local_len) for _ in range(dist.get_world_size())]
            dist.all_gather(world_lens, local_len)
            if self.world_rank == 0 and self.log_to_screen:
                print("Test loader lengths per rank:", [int(t.item()) for t in world_lens])

        # set loss functions
        if self.params.loss_fn == 'MSELoss':
            self.loss_func = torch.nn.MSELoss()
        elif self.params.loss_fn == 'L1Loss':
            self.loss_func = torch.nn.L1Loss()
        elif self.params.loss_fn == 'HuberLoss':
            self.loss_func = torch.nn.HuberLoss()

        self.logs = {}
        
        if self.log_to_screen:
            print("Loading checkpoint %s"%self.params.checkpoint_path)
        self.restore_checkpoint(self.params.checkpoint_path)

        # launch testing
        self.labels, self.predictions, self.visible_energy, self.ve_frac, self.mg_frac, self.oob_frac, self.start_positions, self.rotation_matrices = self.test()
        #print("Start positions:", self.start_positions)
        if self.train_logE == True:
            self.labels = np.exp(self.labels)
            self.predictions = np.exp(self.predictions)
        else:
            self.labels = self.labels*self.params.energy_scaled
            self.predictions = self.predictions*self.params.energy_scaled

       #if dist.is_initialized():
        #    dist.barrier()  # <-- align all ranks following training on one epoch

        for i in range(len(self.labels)):
            if self.world_rank == 0:
                with open(self.params['log_path'], 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([self.labels[i], self.predictions[i], self.visible_energy[i], self.ve_frac[i], self.mg_frac[i], self.oob_frac[i], self.start_positions[i], self.rotation_matrices[i]])
        #self.plot_results()


    def test(self):
        if self.log_to_screen:
            print("Testing the model ...")
        self.model.eval()

        test_start = time.time()

        logs_buff = torch.zeros((1), dtype=torch.float32, device=self.device)
        self.logs['test_loss'] = logs_buff[0].view(-1)

        labels = []
        preds = []
        visible_energy = []
        ve_frac = []
        mg_frac = []
        oob_frac = []
        start_positions = []
        rotation_matrices = []

        # Initialize a progress bar
        pbar = tqdm(total=len(self.test_data_loader), position=0, leave=True)

        with torch.no_grad():
            for i, (inputs, targets, VE_frac, MG_frac, OOB_frac, start_pos, rot_mat) in enumerate(self.test_data_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                VE_frac, MG_frac, OOB_frac, start_pos, rot_mat = VE_frac.to(self.device), MG_frac.to(self.device), OOB_frac.to(self.device), start_pos.to(self.device), rot_mat.to(self.device)
                outputs = self.model(inputs)
                labels.append(targets.detach().reshape(-1))
                preds.append(outputs.detach().reshape(-1))
                ve_frac.append(VE_frac.detach().reshape(-1))
                mg_frac.append(MG_frac.detach().reshape(-1))
                oob_frac.append(OOB_frac.detach().reshape(-1))
                start_positions.append(start_pos.detach())
                rotation_matrices.append(rot_mat.detach())
                #print("Input type: ", type(inputs[0]))
                #print("Start positions:", start_positions[-1])

                loss = self.loss_func(outputs, targets)
                self.logs['test_loss'] += loss.detach() 
                print("Batch {}: Loss = {:.4f}".format(i, loss.item()))
                
                # Get VE 
                batch_ids = inputs[:,0].long()
                visible_energy_values = inputs[:,4]
                assert isinstance(visible_energy_values, torch.Tensor)
                num_batches = int(batch_ids.max().item()) + 1
                visible_energy_sums = torch.zeros(num_batches, device=batch_ids.device)
                visible_energy_sums = visible_energy_sums.scatter_add(0, batch_ids, visible_energy_values)
                visible_energy.append(visible_energy_sums.detach())

                pbar.update(1)

            self.logs['test_loss'] /= len(self.test_data_loader)
            print("Test Loss: {:.4f}".format(self.logs['test_loss'].item()))

        test_time = time.time() - test_start
        if self.log_to_screen:
            print("Test time: {:.2f}s".format(test_time))

        return torch.concat(labels).cpu().numpy(), torch.concat(preds).cpu().numpy(), torch.concat(visible_energy).cpu().numpy(), torch.concat(ve_frac).cpu().numpy(), torch.concat(mg_frac).cpu().numpy(), torch.concat(oob_frac).cpu().numpy(), torch.concat(start_positions).cpu().numpy(), torch.concat(rotation_matrices).cpu().numpy()

    def plot_results(self):

        # Plot true vs. predicted energy w/ visible energy color fraction
        plt.figure(figsize=(10,6))
        plt.scatter(self.labels, self.predictions, c=self.visible_energy/self.labels, cmap='Spectral', vmin=0, alpha=0.7, s=1)
        plt.xlabel('True KE [MeV]')
        plt.ylabel('Predicted Energy [MeV]')
        plt.colorbar(label='Visible Energy Fraction')
        y_eq_x = np.linspace(0,2010, 2010)
        plt.plot(y_eq_x, y_eq_x, 'k' '--', alpha=0.5)
        plt.xlim(-5,2010)
        plt.ylim(-5,3000)
        plt.title('Sheep CNN Energy Prediction')
        plt.savefig(os.path.join(self.params.experiment_dir, 'true_vs_predicted_energy_color_visible_energy_fraction.png'))
        plt.close()

    def restore_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"), weights_only=False) 
        #print("Model state dict keys to start:", self.model.state_dict().keys())
        # exception for handling case where model is wrapped in DDP
        try:
            self.model.load_state_dict(checkpoint['model_state'])
            #print("Loaded model state dict with keys:", checkpoint['model_state'].keys())
        except:
            new_state_dict = OrderedDict()
            for key, val in checkpoint['model_state'].items():
                #print("Model state dict key:", key)
                name = key[7:]
                new_state_dict[name] = val 
            missing, unexpected = self.model.load_state_dict(new_state_dict, strict=False)
            print("Missing keys:", missing)
            print("Unexpected keys:", unexpected)
        
        #print("Model state dict keys: ", self.model.state_dict().keys())

        total = 0
        for p in self.model.parameters():
            total += p.abs().sum().item()

        print("Loaded model weight sum:", total)

        print(f"Restored model weights from {checkpoint_path}")

        
if __name__ == '__main__':
    # parsers for any cmd line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config", default='./configs/default_sheep.yaml', type=str)
    parser.add_argument("--config", default='test', type=str)
    parser.add_argument("--train_config", default='default', type=str)
    parser.add_argument("--results_dir", default='./outputs', type=str, help='directory to store results')
    parser.add_argument("--run_num", default='0', type=str, help='sub run config')
    parser.add_argument("--checkpoint_file", default='ckpt_best.tar', type=str, help='checkpoint file to load')
    args = parser.parse_args()
    params = ParseYAML(os.path.abspath(args.yaml_config), args.config)

    tester = Tester(params, args)
    tester.launch()
    print('Testing complete and results saved to {}'.format(args.results_dir))
