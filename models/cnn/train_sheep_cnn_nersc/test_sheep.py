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

# models
import models.sheep_cnn

class Tester():
    """ trainer class """

    def __init__(self, params, args):

        self.results_dir = args.results_dir
        self.config = args.config 
        self.train_config = args.train_config
        self.run_num = args.run_num
        self.params = params
        self.log_to_screen = 1 # print to screen

        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
        else:
            self.device = torch.device('cpu')

    def init_exp_dir(self, exp_dir, train_dir):
        if not os.path.isdir(exp_dir):
            os.makedirs(exp_dir)
        if not os.path.isdir(train_dir):
            raise ValueError(f"Training directory {train_dir} does not exist. Please train the model before testing.")
        self.params['experiment_dir'] = os.path.abspath(exp_dir)
        self.params['train_dir'] = os.path.abspath(train_dir)
        self.params['checkpoint_path'] = os.path.join(train_dir, 'checkpoints/ckpt_best.tar')
        self.params['resuming'] = True if os.path.isfile(self.params.checkpoint_path) else False

    def launch(self):
        exp_dir = os.path.join(*[self.results_dir, self.config, self.run_num])
        train_dir = os.path.join(*[self.results_dir, self.train_config, self.run_num])
        self.init_exp_dir(exp_dir, train_dir)

        self.params['global_batch_size'] = self.params.batch_size
        self.params['local_batch_size'] = self.params.batch_size
        self.params['global_valid_batch_size'] = self.params.valid_batch_size
        self.params['local_valid_batch_size'] = self.params.valid_batch_size

        # get the dataloaders
        self.test_data_loader, self.test_sampler = get_data_loader(self.params, self.params.test_path, distributed=False, train=False)

        # get the model
        self.model = models.sheep_cnn.sheep_cnn(self.params).to(self.device)

        if self.log_to_screen:
            print("Loading checkpoint %s"%self.params.checkpoint_path)
        self.restore_checkpoint(self.params.checkpoint_path)

        # launch testing
        self.labels, self.predictions, self.visible_energy = self.test()
        self.labels = self.labels*self.params.energy_scaled
        self.predictions = self.predictions*self.params.energy_scaled

        self.plot_results()


    def test(self):
        if self.log_to_screen:
            print("Testing the model ...")
        self.model.eval()

        test_start = time.time()

        logs_buff = torch.zeros((1), dtype=torch.float32, device=self.device)

        labels = []
        preds = []
        visible_energy = []

        # Initialize a progress bar
        pbar = tqdm(total=len(self.test_data_loader), position=0, leave=True)

        with torch.no_grad():
            for i, (inputs, targets, VE_frac, MG_frac, OOB_frac) in enumerate(self.test_data_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                labels.append(targets.detach().reshape(-1))
                preds.append(outputs.detach())
                
                # Get VE 
                batch_ids = inputs[:,0].long()
                visible_energy_values = inputs[:,4]
                assert isinstance(visible_energy_values, torch.Tensor)
                num_batches = int(batch_ids.max().item()) + 1
                visible_energy_sums = torch.zeros(num_batches, device=batch_ids.device)
                visible_energy_sums = visible_energy_sums.scatter_add(0, batch_ids, visible_energy_values)
                visible_energy.append(visible_energy_sums.detach())

                pbar.update(1)

        test_time = time.time() - test_start
        if self.log_to_screen:
            print("Test time: {:.2f}s".format(test_time))

        return torch.concat(labels).cpu().numpy(), torch.concat(preds).cpu().numpy(), torch.concat(visible_energy).cpu().numpy()

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
        checkpoint = torch.load(checkpoint_path, weights_only=True) 
        
        # exception for handling case where model is wrapped in DDP
        try:
            self.model.load_state_dict(checkpoint['model_state'])
        except:
            new_state_dict = OrderedDict()
            for key, val in checkpoint['model_state'].items():
                name = key[7:]
                new_state_dict[name] = val 
            self.model.load_state_dict(new_state_dict)

        print(f"Restored model weights from {checkpoint_path}")

        
if __name__ == '__main__':
    # parsers for any cmd line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config", default='./configs/default_sheep.yaml', type=str)
    parser.add_argument("--config", default='test', type=str)
    parser.add_argument("--train_config", default='default', type=str)
    parser.add_argument("--results_dir", default='./outputs', type=str, help='directory to store results')
    parser.add_argument("--run_num", default='0', type=str, help='sub run config')
    args = parser.parse_args()
    params = ParseYAML(os.path.abspath(args.yaml_config), args.config)

    tester = Tester(params, args)
    tester.launch()
    print('Testing complete and results saved to {}'.format(args.results_dir))
