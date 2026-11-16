import numpy as np
from collections import namedtuple, OrderedDict

import matplotlib
import math
from mpl_toolkits.mplot3d import Axes3D
import os
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/global/cfs/cdirs/dune/users/ehinkle/nd_prototypes_ana/sheep-model/models/cnn/train_sheep_cnn_nersc/utils/')
from utils.parse_yaml import ParseYAML
import json
import h5py
import glob
import os
import argparse
import csv
import ast
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.axes import Axes
from scipy.stats import gaussian_kde
from scipy.optimize import curve_fit, minimize, Bounds, differential_evolution
from mpl_toolkits.mplot3d.axes3d import Axes

def half_sample_mode(data):

    """Computes the half sample mode of a 1D numpy array in the form suggested by Bickel and Frühwirth (2006)"""

    data = np.sort(data)
    n = len(data)

    if n==0:
        raise ValueError("Data array is empty.")
    if n==1:
        return data[0]
    if n==2:
        return np.mean(data)
    if n==3:
        diff1 = data[1] - data[0]
        diff2 = data[2] - data[1]
        if diff1 < diff2:
            return np.mean(data[:2])
        elif diff1 > diff2:
            return np.mean(data[1:])
        else:
            return data[1]
    if n>=4:
        w_min = data[n-1] - data[0]
        i_min = 0
        N = math.ceil(n/2)
        for i in range(n-N+1):
            w = data[i+N-1] - data[i]
            if w < w_min:
                w_min = w
                i_min = i
        return half_sample_mode(data[i_min:i_min+N])

class TestedSheepAll():

    def __init__(self, args):

        self.version = args.version
        self.test_config = args.test_config
        self.train_config = args.train_config
        self.run_num = args.run_num
        self.checkpoint_file = args.checkpoint_file
        self.num_energy_bins = args.num_energy_bins
        self.num_ve_frac_bins = args.num_ve_frac_bins
        self.max_energy = args.max_energy

        self.ckpt_filestring = args.checkpoint_file.replace('.tar', '')

        self.main_dir = args.main_dir
        self.csv_test_file = os.path.join(self.main_dir, self.test_config, self.run_num, 'logs', self.run_num+'_'+self.test_config+'_'+self.ckpt_filestring+args.csv_test_file_end)
        self.get_logs()

        self.color_palette = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', \
                      '#d62728', '#ff9896', '#9467bd', '#c5b0d5', '#8c564b', '#c49c94', \
                      '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d', \
                      '#17becf', '#9edae5']

        self.output_pdf_dir = os.path.join(self.main_dir, self.test_config, self.run_num, 'plots')
        self.output_pdf_base = self.csv_test_file.split('/')[-1].replace('.csv', '')
        self.output_pdf_name = '{}_plot_results_paper.pdf'.format(self.output_pdf_base)
        os.makedirs(self.output_pdf_dir, exist_ok=True)

    def get_logs(self):
        sample_log_dir = self.main_dir+self.train_config+'/'+self.run_num+'/logs/'
        self.csv_loss_log_file = sample_log_dir+self.run_num+'_'+self.train_config+'_train_log.csv'
        self.csv_batch_norm_stats_log_file = sample_log_dir+self.run_num+'_'+self.train_config+'_batch_stats_log.csv'
        #val_pred_log = sample_log_dir+self.run_num+'_'+self.train_config+'_val_pred_log.csv'
        return 0

    def get_values_from_test_csv(self):

        labels = []
        preds = []
        ve = []
        ve_frac = []
        mg_frac = []
        oob_frac = []
        minE = []
        maxE = []
        meanE = []
        medE = []
        numVox = []
        start_position = []
        rotation_matrix = []
        with open(self.csv_test_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                #print(row)
                labels.append(float(row['label']))
                preds.append(float(row['prediction']))
                ve.append(float(row['visible_energy']))        
                ve_frac.append(float(row['ve_frac']))
                mg_frac.append(float(row['mg_frac']))
                oob_frac.append(float(row['oob_frac']))
                minE.append(float(row['minVoxE']))
                maxE.append(float(row['maxVoxE']))
                meanE.append(float(row['meanVoxE']))
                medE.append(float(row['medianVoxE']))
                numVox.append(float(row['numFilledVoxels']))
                start_position.append(str(row['start_position']))
                rotation_matrix.append(str(row['rotation_matrix']))

        self.labels = np.array(labels)
        #print("Labels:", self.labels)
        self.preds = np.array(preds)
        self.ve = np.array(ve)
        self.ve_frac = np.array(ve_frac)
        self.mg_frac = np.array(mg_frac)
        self.oob_frac = np.array(oob_frac)
        self.minE = np.array(minE)
        self.maxE = np.array(maxE)
        self.meanE = np.array(meanE)
        self.medE = np.array(medE)
        self.numVox = np.array(numVox)
        self.start_position = np.array([np.fromstring(s.strip('[]'), sep=' ') for s in start_position])
        self.rotation_matrix = np.array([
            np.fromstring(r.replace('\n', ' ').replace('[', '').replace(']', ''), sep=' ').reshape(3, 3)
                            for r in rotation_matrix])

    def plot_train_val_loss(self, output, ylim=None):

        epochs = []
        val_loss = []
        train_loss = []
        with open(self.csv_loss_log_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                epochs.append(int(row['epoch']))
                val_loss.append(float(row['val_loss']))
                train_loss.append(float(row['train_loss']))

        fig, ax = plt.subplots(figsize=(8,6))
        ax.plot(epochs, val_loss, label='Validation Loss', color=self.color_palette[1], linestyle='--')
        ax.plot(epochs, train_loss, label='Training Loss', color=self.color_palette[0], linestyle='-')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Comparison - ' + self.version)
        ax.legend()
        if ylim is not None:
            ax.set_ylim(ylim)
        #ax.set_yscale('log')
        output.savefig(fig)
        plt.close()

    def plot_batch_vs_running_stats(self, output,layer_name, index=1):

        epochs = []
        global_batch_mean = []
        global_batch_var = []
        running_mean = []
        running_var = []
        with open(self.csv_batch_norm_stats_log_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if str(row['layer_name']) == layer_name:
                    epochs.append(int(row['epoch']))
                    global_batch_mean_list = ast.literal_eval(str(row['global_batch_mean'].split()[1:-2]))
                    global_batch_mean_arr = np.array([float(g) for g in global_batch_mean_list])
                    global_batch_mean.append(float(global_batch_mean_arr[index]))
                    global_batch_var_list = ast.literal_eval(str(row['global_batch_var'].split()[1:-2]))
                    global_batch_var_arr = np.array([float(g) for g in global_batch_var_list])
                    global_batch_var.append(float(global_batch_var_arr[index]))
                    running_mean_list = ast.literal_eval(str(row['running_mean'].split()[1:-2]))
                    running_mean_arr = np.array([float(g) for g in running_mean_list])
                    running_mean.append(float(running_mean_arr[index]))
                    running_var_list = ast.literal_eval(str(row['running_var'].split()[1:-2]))
                    running_var_arr = np.array([float(g) for g in running_var_list])
                    running_var.append(float(running_var_arr[index]))
                else: continue

        fig, ax = plt.subplots(1, 2, figsize=(12,6))
        fig.suptitle(f'Batch vs Running Stats for {layer_name} (Index {index}) - \n' + self.version, fontsize=14)
        ax[0].plot(epochs, global_batch_mean, label='Global Batch Mean', color=self.color_palette[19], linestyle='--')
        ax[0].plot(epochs, running_mean, label='Running Mean', color=self.color_palette[18], linestyle='-')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Mean Value')
        ax[0].set_title(f'Mean Comparison')
        ax[0].legend()      

        ax[1].plot(epochs, global_batch_var, label='Global Batch Variance', color=self.color_palette[19], linestyle='--')
        ax[1].plot(epochs, running_var, label='Running Variance', color=self.color_palette[18], linestyle='-')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Variance Value')
        ax[1].set_title(f'Variance Comparison')
        ax[1].legend()
        output.savefig(fig)
        plt.close()

    def calc_energy_resolution_and_stats(self, estimate, true):

        res = (estimate - true) / true
        #res_gaussian_kde = gaussian_kde(res)
        res_mode = half_sample_mode(res)
        #minimize(lambda x: -res_gaussian_kde(x), x0=np.median(res))
        #res_mode = res_mode_calc.x[0] if res_mode_calc.success else np.nan
        res_scaled = ((res + 1)/(res_mode + 1))-1
        res_abs_scaled = np.abs(res_scaled)
        res_scaled_abs_68th_percentile = np.percentile(res_abs_scaled, 68)

        return res, res_scaled_abs_68th_percentile, res_mode

    
    def plot_energy_res_histogram(self, output):

        # Compute fractional errors and their absolute values and 68th percentile of absolute fractional error
        res_vis_true, res_vis_true_scaled_abs_68th_percentile, res_vis_true_mode = self.calc_energy_resolution_and_stats(self.ve, self.labels)
        res_true_pred, res_true_pred_scaled_abs_68th_percentile, res_true_pred_mode = self.calc_energy_resolution_and_stats(self.preds, self.labels)

        # Compute metrics for plotting energy resolution histograms
        data_min = min(res_true_pred.min(), res_vis_true.min())
        data_max = max(res_true_pred.max(), res_vis_true.max())
        rbins = np.linspace(data_min, data_max, int(round((data_max - data_min), 2) * 50) + 1)
        rbins_centers = (rbins[:-1] + rbins[1:]) / 2
        bin_width = rbins[1] - rbins[0]
        num_events=len(res_vis_true)

        # Compute histograms for both (Visible - True) / True and (SHEEP - True) / True
        hist_counts_vis_true, bin_edges_vis_true = np.histogram(res_vis_true, bins=rbins, density=False)
        #bin_centers_vis_true = (bin_edges_vis_true[:-1] + bin_edges_vis_true[1:]) / 2
        hist_counts_true_pred, bin_edges_true_pred = np.histogram(res_true_pred, bins=rbins, density=False)
        #bin_centers_true_pred = (bin_edges_true_pred[:-1] + bin_edges_true_pred[1:]) / 2
        
        # Construct plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.hist(bin_edges_vis_true[:-1], bins=bin_edges_vis_true, weights=hist_counts_vis_true/ num_events, \
                label="(Visible - True) / True", alpha=0.5, edgecolor="none", color=self.color_palette[8])
        ax.vlines(res_vis_true_mode, ymin=-0.2, ymax=1, color=self.color_palette[8], linestyle=":", linewidth=1)
        ax.hist(bin_edges_true_pred[:-1],bins=bin_edges_true_pred, weights=hist_counts_true_pred/ num_events, \
                label="(SHEEP - True) / True", alpha=0.5, edgecolor="none", color=self.color_palette[12])
        ax.vlines(res_true_pred_mode, ymin=-0.2, ymax=1, color=self.color_palette[12], linestyle=":", linewidth=1)
        ax.vlines(0, ymin=-0.2, ymax=1, color='black', linestyle='--', alpha=0.4, linewidth=1)
        ax.set_ylim(0, 0.11)
        ax.set_xlim(-2, 2)
        ax.legend(fontsize=11)
        ax.set_ylabel(f"Fraction of Test Events / {bin_width:.2f}")
        ax.set_xlabel("Test Event Fractional Error")
        ax.text(-1.89, 0.106, r"$\mathbf{Mode:}$"+f"{res_vis_true_mode:.2f}\n"+r"$\mathbf{Rel. SQAD:}$"+f"{res_vis_true_scaled_abs_68th_percentile:.2f}\n", fontsize=12, verticalalignment='top', color=self.color_palette[8])
        ax.text(-1.89, 0.098, r"$\mathbf{Mode:}$"+f"{res_true_pred_mode:.2f}\n"+r"$\mathbf{Rel. SQAD:}$"+f"{res_true_pred_scaled_abs_68th_percentile:.2f}\n", fontsize=12, verticalalignment='top', color=self.color_palette[12])
        #ax.text(-1.88, 0.20, f"PRELIMINARY", fontsize=18, verticalalignment='top', color='black', alpha=0.35, fontweight='bold')
        ax.text(-1.99, 0.113, self.version, fontsize=12, verticalalignment='top', color='black', alpha=0.8, fontweight='bold')
        output.savefig(fig)
        plt.close()

    def plot_energy_res_histogram_by_energy_bins(self, output, ebins):

        res_vis_true = (self.ve - self.labels)/self.labels
        res_true_pred = (self.preds - self.labels)/self.labels

        # Compute metrics for plotting energy resolution histograms
        data_min = min(res_true_pred.min(), res_vis_true.min())
        data_max = max(res_true_pred.max(), res_vis_true.max())
        rbins = np.linspace(data_min, data_max, int(round((data_max - data_min), 2) * 50) + 1)
        rbin_width = rbins[1] - rbins[0]
        rbins_centers = (rbins[:-1] + rbins[1:]) / 2
        
        ebin_width = ebins[1] - ebins[0]
        true_pred_mode_by_ebin = []
        vis_true_mode_by_ebin = []
        true_pred_scaled_abs_68th_percentile_by_ebin = []
        vis_true_scaled_abs_68th_percentile_by_ebin = []
        num_events_by_ebin = []
        total_events = len(self.labels)

        rows = int(self.num_energy_bins / 5)
        cols = 5
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 6), sharex=False, sharey=False)    
        for i in range(len(ebins)-1):

            row = i // cols
            col = i % cols
            print("Row:", row, "Col:", col)
            bin_mask = (self.labels > ebins[i]) & (self.labels <= ebins[i + 1])
            num_events_in_bin = np.sum(bin_mask)
            if num_events_in_bin == 0:
                true_pred_mode_by_ebin.append(np.nan)
                vis_true_mode_by_ebin.append(np.nan)
                true_pred_scaled_abs_68th_percentile_by_ebin.append(np.nan)
                vis_true_scaled_abs_68th_percentile_by_ebin.append(np.nan)
                num_events_by_ebin.append(0)
                continue
            res_true_pred_ebin, true_pred_scaled_abs_68th_percentile, true_pred_mode= self.calc_energy_resolution_and_stats(self.preds[bin_mask], self.labels[bin_mask])
            res_vis_true_ebin, vis_true_scaled_abs_68th_percentile, vis_true_mode = self.calc_energy_resolution_and_stats(self.ve[bin_mask], self.labels[bin_mask])

            true_pred_mode_by_ebin.append(true_pred_mode)
            true_pred_scaled_abs_68th_percentile_by_ebin.append(true_pred_scaled_abs_68th_percentile)
            vis_true_mode_by_ebin.append(vis_true_mode)
            vis_true_scaled_abs_68th_percentile_by_ebin.append(vis_true_scaled_abs_68th_percentile)
            num_events_by_ebin.append(num_events_in_bin)
            frac_events_by_ebin = num_events_in_bin / total_events *100

            hist_counts_res_vis_true, bin_edges_res_vis_true = np.histogram(res_vis_true_ebin, bins=rbins, density=False)
            hist_counts_res_pred, bin_edges_res_true_pred = np.histogram(res_true_pred_ebin, bins=rbins, density=False)
            axes[row, col].hist(bin_edges_res_vis_true[:-1], bins=bin_edges_res_vis_true, weights=hist_counts_res_vis_true/ num_events_in_bin, \
                            label="(Visible - True) / True", alpha=0.5, edgecolor="none", color=self.color_palette[8])
            axes[row, col].vlines(vis_true_mode, ymin=-0.2, ymax=1, color=self.color_palette[8], linestyle=":", linewidth=1)
            axes[row, col].hist(bin_edges_res_true_pred[:-1], bins=bin_edges_res_true_pred, weights=hist_counts_res_pred/ num_events_in_bin, \
                            label="(SHEEP - True) / True", alpha=0.5, edgecolor="none", color=self.color_palette[12])
            axes[row, col].vlines(true_pred_mode, ymin=-0.2, ymax=1, color=self.color_palette[12], linestyle=":", linewidth=1)
            axes[row, col].vlines(0, ymin=-0.2, ymax=1, color='black', linestyle='--', alpha=0.4, linewidth=1)
            axes[row, col].set_ylim(0, 0.18)
            axes[row, col].set_xlim(-2, 2)
            axes[row, col].legend(fontsize=11)
            axes[row, col].set_ylabel(f"Fraction of Events / {rbin_width:.2f}")
            axes[row, col].set_xlabel("Test Event Fractional Error")
            axes[row, col].text(-1.89, 0.176, r"$\mathbf{Mode:}$"+f"{vis_true_mode:.2f}\n"+r"$\mathbf{Rel. SQAD:}$"+f"{vis_true_scaled_abs_68th_percentile:.2f}\n", fontsize=12, verticalalignment='top', color=self.color_palette[8])
            axes[row, col].text(-1.89, 0.154, r"$\mathbf{Mode:}$"+f"{true_pred_mode:.2f}\n"+r"$\mathbf{Rel. SQAD:}$"+f"{true_pred_scaled_abs_68th_percentile:.2f}\n", fontsize=12, verticalalignment='top', color=self.color_palette[12])
            axes[row, col].text(0.60, 0.8, f"{ebins[i]:.0f}-{ebins[i+1]:.0f} MeV \n{frac_events_by_ebin:.2f}% of Events", transform=axes[row, col].transAxes, fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.2))
        output.savefig(fig)
        plt.close()

    def plot_energy_res_histogram_by_visible_energy_fraction_bins(self, output, ve_frac_bins):

        res_vis_true = (self.ve - self.labels)/self.labels
        res_true_pred = (self.preds - self.labels)/self.labels

        # Compute metrics for plotting energy resolution histograms
        data_min = min(res_true_pred.min(), res_vis_true.min())
        data_max = max(res_true_pred.max(), res_vis_true.max())
        rbins = np.linspace(data_min, data_max, int(round((data_max - data_min), 2) * 50) + 1)
        rbin_width = rbins[1] - rbins[0]
        rbins_centers = (rbins[:-1] + rbins[1:]) / 2
        
        ve_frac_bin_width = ve_frac_bins[1] - ve_frac_bins[0]
        true_pred_mode_by_ve_frac_bin = []
        vis_true_mode_by_ve_frac_bin = []
        true_pred_scaled_abs_68th_percentile_by_ve_frac_bin = []
        vis_true_scaled_abs_68th_percentile_by_ve_frac_bin = []
        num_events_by_ve_frac_bin = []
        total_events = len(self.labels)

        rows = int(self.num_ve_frac_bins / 5)
        cols = 5
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 6), sharex=False, sharey=False)    
        
        for i in range(len(ve_frac_bins)-1):    
            
            row = i // cols
            col = i % cols
            bin_mask = (self.ve_frac > ve_frac_bins[i]) & (self.ve_frac <= ve_frac_bins[i + 1])
            num_events_in_bin = np.sum(bin_mask)
            if num_events_in_bin == 0:
                true_pred_mode_by_ve_frac_bin.append(np.nan)
                vis_true_mode_by_ve_frac_bin.append(np.nan)
                true_pred_scaled_abs_68th_percentile_by_ve_frac_bin.append(np.nan)
                vis_true_scaled_abs_68th_percentile_by_ve_frac_bin.append(np.nan)
                num_events_by_ve_frac_bin.append(0)
                continue
            res_true_pred_ve_frac_bin, true_pred_scaled_abs_68th_percentile, true_pred_mode = self.calc_energy_resolution_and_stats(self.preds[bin_mask], self.labels[bin_mask])
            res_vis_true_ve_frac_bin, vis_true_scaled_abs_68th_percentile, vis_true_mode = self.calc_energy_resolution_and_stats(self.ve[bin_mask], self.labels[bin_mask])

            true_pred_mode_by_ve_frac_bin.append(true_pred_mode)
            true_pred_scaled_abs_68th_percentile_by_ve_frac_bin.append(true_pred_scaled_abs_68th_percentile)
            vis_true_mode_by_ve_frac_bin.append(vis_true_mode)
            vis_true_scaled_abs_68th_percentile_by_ve_frac_bin.append(vis_true_scaled_abs_68th_percentile)
            num_events_by_ve_frac_bin.append(num_events_in_bin)
            frac_events_by_ve_frac_bin = num_events_in_bin / total_events *100
            print("Bin:", ve_frac_bins[i], "-", ve_frac_bins[i+1])
            print("Num Events in Visible Energy Fraction Bin:", num_events_in_bin, "Fraction of Total Events:", frac_events_by_ve_frac_bin)
            print("Total Events:", total_events)

            hist_counts_res_vis_true, bin_edges_res_vis_true = np.histogram(res_vis_true_ve_frac_bin, bins=rbins, density=False)
            hist_counts_res_pred, bin_edges_res_true_pred = np.histogram(res_true_pred_ve_frac_bin, bins=rbins, density=False)
            axes[row, col].hist(bin_edges_res_vis_true[:-1], bins=bin_edges_res_vis_true, weights=hist_counts_res_vis_true/ num_events_in_bin, \
                            label="(Visible - True) / True", alpha=0.5, edgecolor="none", color=self.color_palette[8])
            axes[row, col].vlines(vis_true_mode, ymin=-0.2, ymax=1, color=self.color_palette[8], linestyle=":", linewidth=1)
            axes[row, col].hist(bin_edges_res_true_pred[:-1], bins=bin_edges_res_true_pred, weights=hist_counts_res_pred/ num_events_in_bin, \
                            label="(SHEEP - True) / True", alpha=0.5, edgecolor="none", color=self.color_palette[12])
            axes[row, col].vlines(true_pred_mode, ymin=-0.2, ymax=1, color=self.color_palette[12], linestyle=":", linewidth=1)
            axes[row, col].vlines(0, ymin=-0.2, ymax=1, color='black', linestyle='--', alpha=0.4, linewidth=1)
            axes[row, col].set_ylim(0, 0.18)
            axes[row, col].set_xlim(-2, 2)
            axes[row, col].legend(fontsize=11)
            axes[row, col].set_ylabel(f"Fraction of Events / {rbin_width:.2f}")
            axes[row, col].set_xlabel("Test Event Fractional Error")
            axes[row, col].text(-1.89, 0.176, r"$\mathbf{Mode:}$"+f"{vis_true_mode:.2f}\n"+r"$\mathbf{Rel. SQAD:}$"+f"{vis_true_scaled_abs_68th_percentile:.2f}\n", fontsize=12, verticalalignment='top', color=self.color_palette[8])
            axes[row, col].text(-1.89, 0.154, r"$\mathbf{Mode:}$"+f"{true_pred_mode:.2f}\n"+r"$\mathbf{Rel. SQAD:}$"+f"{true_pred_scaled_abs_68th_percentile:.2f}\n", fontsize=12, verticalalignment='top', color=self.color_palette[12])
            axes[row, col].text(0.60, 0.8, f"{ve_frac_bins[i]:.2f}-{ve_frac_bins[i+1]:.2f} \n{frac_events_by_ve_frac_bin:.2f}% of Events", transform=axes[row, col].transAxes, fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.2))
        output.savefig(fig)
        plt.close()
   
    def plot_energy_res_mode_width_by_energy_bins(self, output, ebins):

        bin_width = ebins[1] - ebins[0]
        true_pred_mode_by_ebin = []
        vis_true_mode_by_ebin = []
        true_pred_scaled_abs_68th_percentile_by_ebin = []
        vis_true_scaled_abs_68th_percentile_by_ebin = []
        ebin_centers = 0.5 * (ebins[:-1] + ebins[1:])
        for i in range(len(ebins)-1):
            bin_mask = (self.labels > ebins[i]) & (self.labels <= ebins[i + 1])
            if np.sum(bin_mask) == 0:
                true_pred_mode_by_ebin.append(np.nan)
                vis_true_mode_by_ebin.append(np.nan)
                true_pred_scaled_abs_68th_percentile_by_ebin.append(np.nan)
                vis_true_scaled_abs_68th_percentile_by_ebin.append(np.nan)
                continue
            _, true_pred_scaled_abs_68th_percentile, true_pred_mode = self.calc_energy_resolution_and_stats(self.preds[bin_mask], self.labels[bin_mask])
            _, vis_true_scaled_abs_68th_percentile, vis_true_mode = self.calc_energy_resolution_and_stats(self.ve[bin_mask], self.labels[bin_mask])

            true_pred_mode_by_ebin.append(true_pred_mode)
            true_pred_scaled_abs_68th_percentile_by_ebin.append(true_pred_scaled_abs_68th_percentile)
            vis_true_mode_by_ebin.append(vis_true_mode)
            vis_true_scaled_abs_68th_percentile_by_ebin.append(vis_true_scaled_abs_68th_percentile)

        fig, ax = plt.subplots(1,2,figsize=(17,6))
        ax[0].scatter(ebin_centers, true_pred_mode_by_ebin, color=self.color_palette[12], label="SHEEP", marker='*')
        ax[0].scatter(ebin_centers, vis_true_mode_by_ebin, color=self.color_palette[8], label="Visible Energy", marker='x')
        ax[0].set_xlabel("True Event Energy [MeV]")
        ax[0].set_ylabel(f"Mode of Fractional Error / {bin_width:.2f} MeV")
        ax[0].set_title(f"Mode of Fractional Error by True Energy Bin - \n{self.version}")
        ax[0].legend()

        ax[1].scatter(ebin_centers, true_pred_scaled_abs_68th_percentile_by_ebin, color=self.color_palette[12], label="SHEEP", marker='*')
        ax[1].scatter(ebin_centers, vis_true_scaled_abs_68th_percentile_by_ebin, color=self.color_palette[8], label="Visible Energy", marker='x')
        ax[1].set_xlabel("True Event Energy [MeV]")
        ax[1].set_ylabel(f"Relative Standard Quantile Absolute Deviation / {bin_width:.2f} MeV")
        ax[1].set_title(f"Relative Standard Quantile Absolute Deviation by True Energy Bin - \n{self.version}")
        ax[1].legend()
        print("Energy bins:", ebins)
        print("True pred mode by ebin:", true_pred_mode_by_ebin)
        print("True pred abs 68th percentile by ebin:", true_pred_scaled_abs_68th_percentile_by_ebin)
        print("Vis true mode by ebin:", vis_true_mode_by_ebin)
        print("Vis true abs 68th percentile by ebin:", vis_true_scaled_abs_68th_percentile_by_ebin)
        for a in ax:
            a.xaxis.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.3)
            a.yaxis.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.2)
        output.savefig(fig)
        plt.close()

    def plot_energy_res_mode_width_by_visible_energy_fraction_bins(self, output, ve_frac_bins):

        bin_width = ve_frac_bins[1] - ve_frac_bins[0]
        true_pred_mode_by_ve_frac_bin = []
        vis_true_mode_by_ve_frac_bin = []
        true_pred_scaled_abs_68th_percentile_by_ve_frac_bin = []
        vis_true_scaled_abs_68th_percentile_by_ve_frac_bin = []
        ve_frac_bin_centers = 0.5 * (ve_frac_bins[:-1] + ve_frac_bins[1:])
        for i in range(len(ve_frac_bins)-1):
            bin_mask = (self.ve_frac > ve_frac_bins[i]) & (self.ve_frac <= ve_frac_bins[i + 1])
            if np.sum(bin_mask) == 0:
                true_pred_mode_by_ve_frac_bin.append(np.nan)
                vis_true_mode_by_ve_frac_bin.append(np.nan)
                true_pred_scaled_abs_68th_percentile_by_ve_frac_bin.append(np.nan)
                vis_true_scaled_abs_68th_percentile_by_ve_frac_bin.append(np.nan)
                continue
            _, true_pred_scaled_abs_68th_percentile, true_pred_mode = self.calc_energy_resolution_and_stats(self.preds[bin_mask], self.labels[bin_mask])
            _, vis_true_scaled_abs_68th_percentile, vis_true_mode = self.calc_energy_resolution_and_stats(self.ve[bin_mask], self.labels[bin_mask])

            true_pred_mode_by_ve_frac_bin.append(true_pred_mode)
            true_pred_scaled_abs_68th_percentile_by_ve_frac_bin.append(true_pred_scaled_abs_68th_percentile)
            vis_true_mode_by_ve_frac_bin.append(vis_true_mode)
            vis_true_scaled_abs_68th_percentile_by_ve_frac_bin.append(vis_true_scaled_abs_68th_percentile)

        fig, ax = plt.subplots(1,2,figsize=(17,6))
        ax[0].scatter(ve_frac_bin_centers, true_pred_mode_by_ve_frac_bin, color=self.color_palette[12], label="SHEEP", marker='*')
        ax[0].scatter(ve_frac_bin_centers, vis_true_mode_by_ve_frac_bin, color=self.color_palette[8], label="Visible Energy", marker='x')
        ax[0].set_xlabel("True Visible Energy Fraction")
        ax[0].set_ylabel(f"Mode of Fractional Error / {bin_width:.2f}")
        ax[0].set_title(f"Mode of Fractional Error by Visible Energy Fraction Bin - \n{self.version}")
        ax[0].legend()

        ax[1].scatter(ve_frac_bin_centers, true_pred_scaled_abs_68th_percentile_by_ve_frac_bin, color=self.color_palette[12], label="SHEEP", marker='*')
        ax[1].scatter(ve_frac_bin_centers, vis_true_scaled_abs_68th_percentile_by_ve_frac_bin, color=self.color_palette[8], label="Visible Energy", marker='x')
        ax[1].set_xlabel("True Visible Energy Fraction")
        ax[1].set_ylabel(f"Relative Standard Quantile Absolute Deviation / {bin_width:.2f} MeV")
        ax[1].set_title(f"Relative Standard Quantile Absolute Deviation by Visible Energy Fraction Bin - \n{self.version}")
        ax[1].legend()
        print("Visible energy fraction bins:", ve_frac_bins)
        print("True pred mode by ve frac bin:", true_pred_mode_by_ve_frac_bin)
        print("True pred abs 68th percentile by ve frac bin:", true_pred_scaled_abs_68th_percentile_by_ve_frac_bin)
        print("Vis true mode by ve frac bin:", vis_true_mode_by_ve_frac_bin)
        print("Vis true abs 68th percentile by ve frac bin:", vis_true_scaled_abs_68th_percentile_by_ve_frac_bin)
        
        for a in ax:
            a.xaxis.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.3)
            a.yaxis.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.2)
        output.savefig(fig)
        plt.close()

    def make_plots(self):

        # Rasterize plots 
        _old_axes_init = Axes.__init__
        def _new_axes_init(self, *a, **kw):
            _old_axes_init(self, *a, **kw)
            # https://matplotlib.org/stable/gallery/misc/zorder_demo.html
            # 3 => leave text and legends vectorized
            self.set_rasterization_zorder(3)
        def rasterize_plots():
            Axes.__init__ = _new_axes_init
        def vectorize_plots():
            Axes.__init__ = _old_axes_init


        with PdfPages(os.path.join(self.output_pdf_dir, self.output_pdf_name), keep_empty=False) as output:

            # Set plot parameters
            plt.rcParams.update({'font.size': 12})
            plt.rcParams.update({'legend.fontsize': 12})
            #plt.rcParams['font.family'] = 'serif'
            #plt.rcParams['mathtext.fontset'] = 'stix'
            plt.rcParams['xtick.direction'] = 'in'
            plt.rcParams['ytick.direction'] = 'in'
            plt.rcParams['xtick.top'] = True
            plt.rcParams['ytick.right'] = True
            plt.rcParams['xtick.major.size'] = 5
            plt.rcParams['xtick.minor.size'] = 3
            plt.rcParams['ytick.major.size'] = 5
            plt.rcParams['ytick.minor.size'] = 3

            # Set up bins for visible energy fraction and energy
            self.ve_frac_bins = np.linspace(0, 2, self.num_ve_frac_bins + 1)
            self.energy_bins = np.linspace(0, self.max_energy, self.num_energy_bins + 1)

            # Plot the training and validation loss curves
            #self.plot_train_val_loss(output)

            # Plot a bunch of running stats in an early layer and a later layer
            #self.plot_batch_vs_running_stats(output, layer_name='encoding_block.2.1.norm2', index=0)
            #self.plot_batch_vs_running_stats(output, layer_name='encoding_block.2.1.norm2', index=1)
            #self.plot_batch_vs_running_stats(output, layer_name='encoding_block.2.1.norm2', index=2)
            #self.plot_batch_vs_running_stats(output, layer_name='encoding_block.2.1.norm2', index=3)
            #self.plot_batch_vs_running_stats(output, layer_name='encoding_block.2.1.norm2', index=4)
            #self.plot_batch_vs_running_stats(output, layer_name='encoding_block.2.1.norm2', index=5)
            #self.plot_batch_vs_running_stats(output, layer_name='encoding_block.8.0.norm1', index=10)
            #self.plot_batch_vs_running_stats(output, layer_name='encoding_block.8.0.norm1', index=11)
            #self.plot_batch_vs_running_stats(output, layer_name='encoding_block.8.0.norm1', index=12)
            #self.plot_batch_vs_running_stats(output, layer_name='encoding_block.8.0.norm1', index=13)
            #self.plot_batch_vs_running_stats(output, layer_name='encoding_block.8.0.norm1', index=14)

            # Plot energy resolution histograms for both (Visible - True) / True and (SHEEP - True) / True
            self.plot_energy_res_histogram(output)
            self.plot_energy_res_mode_width_by_energy_bins(output, self.energy_bins)
            self.plot_energy_res_histogram_by_energy_bins(output, self.energy_bins)
            self.plot_energy_res_mode_width_by_visible_energy_fraction_bins(output, self.ve_frac_bins)
            self.plot_energy_res_histogram_by_visible_energy_fraction_bins(output, self.ve_frac_bins)



            '''#### OOB Frac vs. MG Frac by visible energy fraction bin
            self.missing_frac_bins = np.linspace(0, 1, 20 + 1)

            fig, ax = plt.subplots(2,5, figsize=(38, 14), sharex=False, sharey=False)
            ax = ax.flatten()

            im = None
    
            for i in range(self.num_ve_frac_bins):
                bin_mask = (self.ve / self.labels > self.ve_frac_bins[i]) & (self.ve / self.labels <= self.ve_frac_bins[i + 1])
                if np.sum(bin_mask) == 0:
                    continue
                h = ax[i].hist2d(
                    self.oob_frac[bin_mask],
                    self.mg_frac[bin_mask],
                    bins=self.missing_frac_bins,
                    cmap='magma_r',
                    cmin=1,
                )
                im = h[3]  # <-- grab the QuadMesh (image)
                ax[i].set_xlabel('Uncontained Energy Fraction', fontsize=16)
                ax[i].set_ylabel('Module Gap Energy Fraction', fontsize=16)
                #ax[i].legend(loc='upper right', fontsize=16)
                ax[i].tick_params(axis='both', which='major', labelsize=14)

                frac_events = round(len(self.preds[bin_mask])/len(self.preds), 4)*100
                ax[i].text(0.5, 0.97, f'VE Fraction: {self.ve_frac_bins[i]:.2f}-{self.ve_frac_bins[i+1]:.2f}\n{frac_events:.2f}% of Events', 
                           transform=ax[i].transAxes, 
                           color='rebeccapurple', 
                           fontsize=15, 
                           verticalalignment='top', 
                           bbox=dict(boxstyle='round', 
                                     facecolor='white', alpha=0.8))
            # Add ONE shared colorbar on the right
            cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.02, pad=0.02)
            cbar.set_label('Events', fontsize=16)
            fig.suptitle(f'{self.version} Uncontained vs. Module Gap Missing Energy by Visible Energy Fraction Bin', size=20)
            #fig.tight_layout()
            output.savefig(fig)
            plt.close()

            #### OOB Frac vs. MG Frac by visible energy fraction bin
            fig, ax = plt.subplots(figsize=(6,4))
            h = plt.hist2d(self.labels,self.oob_frac,bins=(self.ebins,self.missing_frac_bins),cmap='magma_r',cmin=1)
            im = h[3]
            plt.xlabel("True Event Energy [MeV]")
            plt.ylabel("Uncontained Energy Fraction")
            cbar = fig.colorbar(im, ax=ax,orientation='vertical', fraction=0.02, pad=0.02)
            cbar.set_label('Events', fontsize=12)
            fig.suptitle(f'{self.version} Uncontained Missing Energy by True Event Energy', size=14)
            output.savefig(fig)
            plt.close()



            #### Energy resolution by uncontained and module gap fraction double bins
            self.oob_mg_frac_bins = np.linspace(0, 1, 5 + 1)
            self.res_true_pred_energy = (self.preds - self.labels) / self.labels
            self.res_true_vis_energy = (self.ve - self.labels) / self.labels

            def plot_double_binned_resolution(residuals_pred, residuals_vis, fig_title):
                nbins = len(self.oob_mg_frac_bins) - 1
                fig, ax = plt.subplots(nbins, nbins, figsize=(4.5 * nbins, 4.0 * nbins), sharex=True, sharey=True)
                if nbins == 1:
                    ax = np.array([[ax]])

                finite_pred = residuals_pred[np.isfinite(residuals_pred)]
                finite_vis = residuals_vis[np.isfinite(residuals_vis)]
                if finite_pred.size == 0 and finite_vis.size == 0:
                    plt.close(fig)
                    return None, None, None, None, None, None

                finite_all = np.concatenate([finite_pred, finite_vis])
                res_min = finite_all.min()
                res_max = finite_all.max()
                if np.isclose(res_min, res_max):
                    res_min -= 0.5
                    res_max += 0.5
                pad = 0.05 * (res_max - res_min)
                rbins = np.linspace(res_min - pad, res_max + pad, 61)
                bin_width = rbins[1] - rbins[0]

                mean_grid_pred = np.full((nbins, nbins), np.nan)
                std_grid_pred = np.full((nbins, nbins), np.nan)
                count_grid_pred = np.zeros((nbins, nbins), dtype=int)
                mean_grid_vis = np.full((nbins, nbins), np.nan)
                std_grid_vis = np.full((nbins, nbins), np.nan)
                count_grid_vis = np.zeros((nbins, nbins), dtype=int)

                for i_oob in range(nbins):
                    oob_low = self.oob_mg_frac_bins[i_oob]
                    oob_high = self.oob_mg_frac_bins[i_oob + 1]
                    if i_oob == nbins - 1:
                        oob_mask = (self.oob_frac >= oob_low) & (self.oob_frac <= oob_high)
                    else:
                        oob_mask = (self.oob_frac >= oob_low) & (self.oob_frac < oob_high)

                    for j_mg in range(nbins):
                        mg_low = self.oob_mg_frac_bins[j_mg]
                        mg_high = self.oob_mg_frac_bins[j_mg + 1]
                        if j_mg == nbins - 1:
                            mg_mask = (self.mg_frac >= mg_low) & (self.mg_frac <= mg_high)
                        else:
                            mg_mask = (self.mg_frac >= mg_low) & (self.mg_frac < mg_high)

                        bin_mask_pred = oob_mask & mg_mask & np.isfinite(residuals_pred)
                        bin_mask_vis = oob_mask & mg_mask & np.isfinite(residuals_vis)
                        n_events_pred = int(np.count_nonzero(bin_mask_pred))
                        n_events_vis = int(np.count_nonzero(bin_mask_vis))
                        count_grid_pred[i_oob, j_mg] = n_events_pred
                        count_grid_vis[i_oob, j_mg] = n_events_vis

                        if n_events_pred == 0 and n_events_vis == 0:
                            ax[i_oob, j_mg].axis('off')
                            continue

                        res_bin_pred = residuals_pred[bin_mask_pred]
                        res_bin_vis = residuals_vis[bin_mask_vis]
                        total_events = len(finite_pred)

                        if n_events_pred > 0:
                            ax[i_oob, j_mg].hist(
                                res_bin_pred,
                                bins=rbins,
                                weights=np.ones_like(res_bin_pred) / n_events_pred,
                                color='sienna',
                                alpha=0.60,
                                edgecolor='none',
                                label='Predicted energy',
                            )
                            mean_grid_pred[i_oob, j_mg] = np.mean(res_bin_pred)
                            std_grid_pred[i_oob, j_mg] = np.std(res_bin_pred)

                        if n_events_vis > 0:
                            ax[i_oob, j_mg].hist(
                                res_bin_vis,
                                bins=rbins,
                                weights=np.ones_like(res_bin_vis) / n_events_vis,
                                color='royalblue',
                                alpha=0.45,
                                edgecolor='none',
                                label='Visible energy',
                            )
                            mean_grid_vis[i_oob, j_mg] = np.mean(res_bin_vis)
                            std_grid_vis[i_oob, j_mg] = np.std(res_bin_vis)

                        ax[i_oob, j_mg].axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.7)
                        if n_events_pred > 0:
                            ax[i_oob, j_mg].axvline(mean_grid_pred[i_oob, j_mg], color='sienna', linewidth=1.0)
                        if n_events_vis > 0:
                            ax[i_oob, j_mg].axvline(mean_grid_vis[i_oob, j_mg], color='royalblue', linewidth=1.0)

                        frac_events_pred = 100.0 * n_events_pred / total_events
                        frac_events_vis = 100.0 * n_events_vis / total_events
                        text_lines = [f'OOB {oob_low:.2f}-{oob_high:.2f}', f'MG {mg_low:.2f}-{mg_high:.2f}']
                        if n_events_pred > 0:
                            text_lines.append(f'Pred N={n_events_pred} ({frac_events_pred:.2f}%)')
                            text_lines.append(f'Pred mean: {mean_grid_pred[i_oob, j_mg]:.2f}')
                            text_lines.append(f'Pred std: {std_grid_pred[i_oob, j_mg]:.2f}')
                        if n_events_vis > 0:
                            text_lines.append(f'Vis N={n_events_vis} ({frac_events_vis:.2f}%)')
                            text_lines.append(f'Vis mean: {mean_grid_vis[i_oob, j_mg]:.2f}')
                            text_lines.append(f'Vis std: {std_grid_vis[i_oob, j_mg]:.2f}')
                        ax[i_oob, j_mg].text(
                            0.03,
                            0.97,
                            '\n'.join(text_lines),
                            transform=ax[i_oob, j_mg].transAxes,
                            fontsize=9,
                            verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                        )

                        if i_oob == nbins - 1:
                            ax[i_oob, j_mg].set_xlabel('Event Energy Resolution', fontsize=11)
                        if j_mg == 0:
                            ax[i_oob, j_mg].set_ylabel(f'Fraction / {bin_width:.2f}', fontsize=11)
                        ax[i_oob, j_mg].tick_params(axis='both', which='major', labelsize=10)
                        if n_events_pred > 0 or n_events_vis > 0:
                            ax[i_oob, j_mg].legend(loc='upper right', fontsize=8)

                fig.suptitle(fig_title, size=20)
                fig.tight_layout(rect=[0, 0, 1, 0.96])
                output.savefig(fig)
                plt.close(fig)
                return (
                    mean_grid_pred,
                    std_grid_pred,
                    count_grid_pred,
                    mean_grid_vis,
                    std_grid_vis,
                    count_grid_vis,
                )

            (
                self.res_true_pred_energy_mean_by_oob_mg_bin,
                self.res_true_pred_energy_std_by_oob_mg_bin,
                self.res_true_pred_energy_num_events_by_oob_mg_bin,
                self.res_true_vis_energy_mean_by_oob_mg_bin,
                self.res_true_vis_energy_std_by_oob_mg_bin,
                self.res_true_vis_energy_num_events_by_oob_mg_bin,
            ) = plot_double_binned_resolution(
                self.res_true_pred_energy,
                self.res_true_vis_energy,
                f'{self.version} Predicted Energy Resolution by Uncontained and Module Gap Fractions',
            )

            #### Plot Voxel Metrics (Total)
            print("Starting voxel metric plots...")

            # True Energy vs. Num Voxels by VE Frac -- too long to produce for now
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(self.labels, self.numVox, c=self.ve/self.labels, cmap='Spectral', vmin=0, alpha=0.7, s=1)
            ax.plot([self.labels.min(), self.labels.max()], [self.labels.min(), self.labels.max()], 'r--')
            ax.set_xlabel('True KE [MeV]')
            ax.set_ylabel('Number of Filled Voxels')
            ax.set_title(f'{self.version} True Energy vs. Number of Filled Voxels')
            plt.colorbar(ax.collections[0], label='Visible Energy Fraction')
            ax.set_xlim(-5,15010)
            ax.legend()
            output.savefig(fig)
            plt.close(fig)

            # Number of filled voxels histogram (Total)
            fig, ax = plt.subplots(figsize=(22, 8))
            num_events=len(self.numVox)
            #print("Number of events:", num_events)
            numVox_bins = np.linspace(self.numVox.min(), self.numVox.max(), int(round((self.numVox.max() - self.numVox.min()), 1)/100) + 1)
            #print("Number of voxels -- bins:", numVox_bins)
            numVox_bin_width = numVox_bins[1] - numVox_bins[0]
            hist_counts_numVox, bin_edges_numVox = np.histogram(self.numVox, bins=numVox_bins, density=False)
            ax.hist(bin_edges_numVox[:-1], bins=bin_edges_numVox, weights=hist_counts_numVox/num_events, label="Number of Voxels", alpha=0.5, edgecolor="none")
            ax.legend(fontsize=11)
            ax.set_ylabel(f"Fraction of Test Events / {numVox_bin_width:.2f}")
            ax.set_xlabel("Test Event Number of Voxels Filled")
            ax=plt.gca()
            ax.text(-1.88, 0.267, self.version, fontsize=12, verticalalignment='top', color='black', alpha=0.8, fontweight='bold')
            fig.tight_layout()
            output.savefig(fig)
            plt.close()

            # Minimum Voxel Energy histogram (Total)
            fig, ax = plt.subplots(figsize=(8, 6))
            num_events=len(self.minE)
            minE_bins = np.linspace(self.minE.min(), self.minE.max(), 1001)
            print("Unique minE:", np.unique(self.minE))
            if self.minE.min() == self.minE.max():
                minE_bins = np.linspace(self.minE.min(), self.minE.max()+1, 1001)
            print("Minimum Voxel Energy -- bins:", minE_bins)
            minE_bin_width = minE_bins[1] - minE_bins[0]
            hist_counts_minE, bin_edges_minE = np.histogram(self.minE, bins=minE_bins, density=False)
            ax.hist(bin_edges_minE[:-1], bins=bin_edges_minE, weights=hist_counts_minE/num_events, label="Minimum Voxel Energy [MeV]", alpha=0.5, edgecolor="none")
            ax.legend(fontsize=11)
            ax.set_ylabel(f"Fraction of Test Events / {minE_bin_width:.2f} [MeV]")
            ax.set_xlabel("Test Event Minimum Voxel Energy [MeV]")
            ax=plt.gca()
            ax.text(-1.88, 0.267, self.version, fontsize=12, verticalalignment='top', color='black', alpha=0.8, fontweight='bold')
            fig.tight_layout()
            output.savefig(fig)
            plt.close()

            # Maximum Voxel Energy histogram (Total)
            fig, ax = plt.subplots(figsize=(8, 6))
            num_events=len(self.maxE)
            maxE_bins = np.linspace(self.maxE.min(), self.maxE.max(), int(round((self.maxE.max() - self.maxE.min()), 1) * 10) + 1)
            maxE_bin_width = maxE_bins[1] - maxE_bins[0]
            hist_counts_maxE, bin_edges_maxE = np.histogram(self.maxE, bins=maxE_bins, density=False)
            ax.hist(bin_edges_maxE[:-1], bins=bin_edges_maxE, weights=hist_counts_maxE/num_events, label="Maximum Voxel Energy [MeV]", alpha=0.5, edgecolor="none")
            ax.legend(fontsize=11)
            ax.set_ylabel(f"Fraction of Test Events / {maxE_bin_width:.2f} [MeV]")
            ax.set_xlabel("Test Event Maximum Voxel Energy [MeV]")
            ax=plt.gca()
            ax.text(-1.88, 0.267, self.version, fontsize=12, verticalalignment='top', color='black', alpha=0.8, fontweight='bold')
            fig.tight_layout()
            output.savefig(fig)
            plt.close()

            # Mean and Median Voxel Energy histograms (Total)
            fig, ax = plt.subplots(figsize=(8, 6))
            num_events=len(self.meanE)
            avgE_min = min(self.meanE.min(), self.medE.min())
            avgE_max = min(self.meanE.max(), self.medE.max())
            avgE_bins = np.linspace(avgE_min, avgE_max, int(round((avgE_max - avgE_min), 1) * 100) + 1)
            avgE_bin_width = avgE_bins[1] - avgE_bins[0]
            hist_counts_meanE, bin_edges_meanE = np.histogram(self.meanE, bins=avgE_bins, density=False)
            hist_counts_medE, bin_edges_medE = np.histogram(self.medE, bins=avgE_bins, density=False)
            ax.hist(bin_edges_meanE[:-1], bins=bin_edges_meanE, weights=hist_counts_meanE/num_events, label="Mean Voxel Energy [MeV]", alpha=0.5, edgecolor="none")
            ax.hist(bin_edges_medE[:-1], bins=bin_edges_medE, weights=hist_counts_medE/num_events, label="Median Voxel Energy [MeV]", alpha=0.5, edgecolor="none")
            ax.legend(fontsize=11)
            ax.set_ylabel(f"Fraction of Test Events / {avgE_bin_width:.2f} [MeV]")
            ax.set_xlabel("Test Event Average Voxel Energy [MeV]")
            ax=plt.gca()
            ax.text(-1.88, 0.267, self.version, fontsize=12, verticalalignment='top', color='black', alpha=0.8, fontweight='bold')
            fig.tight_layout()
            output.savefig(fig)
            plt.close()

            print("Starting histograms by energy bin for voxel metrics...")

            def plot_single_voxel_metric_by_energy_bin(vox_metric=self.minE, vox_metric_name="Minimum Voxel Energy"):
    
                self.num_events_by_ebin = []
                ebin_centers = 0.5 * (self.ebins[:-1] + self.ebins[1:])
                if vox_metric_name == 'Minimum Voxel Energy':
                    rbins = np.linspace(self.minE.min(), self.minE.max(), 1001)
                    if self.minE.min() == self.minE.max():
                        rbins = np.linspace(self.minE.min(), self.minE.max()+1, 1001)
                elif vox_metric_name == 'Maximum Voxel Energy':
                    rbins = np.linspace(self.maxE.min(), self.maxE.max(), int(round((self.maxE.max() - self.maxE.min()), 1)*10) + 1)
                elif vox_metric_name == 'Number of Filled Voxels':
                    rbins = np.linspace(self.numVox.min(), self.numVox.max(), int(round((self.numVox.max() - self.numVox.min()), 1)/100) + 1)
                else: 
                    print(f"Vox metric name of {vox_metric_name} unknown ... function is breaking.")
                    return
                bin_width = rbins[1] - rbins[0]

                fig, ax = plt.subplots(int(self.num_energy_bins/5),5, figsize=(30, int(6*(self.num_energy_bins/5))))#, sharex=True, sharey=True)
                ax = ax.flatten()

                for i in range(self.num_energy_bins):
                    bin_mask = (self.labels > self.ebins[i]) & (self.labels <= self.ebins[i + 1])
                    if np.sum(bin_mask) == 0:
                        continue
                    vox_metric_bin = vox_metric[bin_mask]
                    num_events_per_energy_bin = len(vox_metric_bin)
                    frac_events_per_energy_bin = (num_events_per_energy_bin/len(vox_metric))*100
                    
                    hist_counts_vox_metric, bin_edges_vox_metric = np.histogram(vox_metric_bin, bins=rbins, density=False)
                    ax[i].hist(bin_edges_vox_metric[:-1], bins=bin_edges_vox_metric, weights=hist_counts_vox_metric/num_events_per_energy_bin, label=f'{vox_metric_name} [MeV]', alpha=0.5, edgecolor="none")
                    ax[i].set_xlabel(f'{vox_metric_name} [MeV]', fontsize=16)
                    ax[i].set_ylabel(f'Fraction of Test Events / {round(bin_width, 2)} [MeV]', fontsize=16)
                    ax[i].legend(loc='upper right', fontsize=16)
                    ax[i].tick_params(axis='both', which='major', labelsize=14)
                    ax[i].text(0.63, 0.8, f"{self.ebins[i]:.0f}-{self.ebins[i+1]:.0f} MeV \n{frac_events_per_energy_bin:.2f}% of Events", transform=ax[i].transAxes, fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', color='orange', alpha=0.2))
                    ax[i].legend(loc='upper right', fontsize=16)
                    self.num_events_by_ebin.append(num_events_per_energy_bin)

                fig.suptitle(f'{vox_metric_name} by True Energy Bin for {self.version}', size=20)
                fig.tight_layout()
                output.savefig(fig)
                self.num_events_by_ebin = np.array(self.num_events_by_ebin)
                plt.close()
            plot_single_voxel_metric_by_energy_bin(self.minE, "Minimum Voxel Energy")
            plot_single_voxel_metric_by_energy_bin(self.maxE, "Maximum Voxel Energy")
            plot_single_voxel_metric_by_energy_bin(self.numVox, "Number of Filled Voxels")
            print("Starting histograms by visible energy fraction bin for voxel metrics...")
            def plot_avg_voxel_metrics_by_energy_bin(vox_metric1=self.meanE, vox_metric_name1="Mean Voxel Energy", vox_metric2=self.medE, vox_metric_name2="Median Voxel Energy"):
    
                self.num_events_by_ebin = []
                ebin_centers = 0.5 * (self.ebins[:-1] + self.ebins[1:])
                vox_metric_min = min(vox_metric1.min(), vox_metric2.min())
                vox_metric_max = min(vox_metric1.max(), vox_metric2.max())
                rbins = np.linspace(vox_metric_min, vox_metric_max, int(round((vox_metric_max - vox_metric_min), 1) * 100) + 1)
                bin_width = rbins[1] - rbins[0]

                fig, ax = plt.subplots(int(self.num_energy_bins/5),5, figsize=(30, int(6*(self.num_energy_bins/5))))#, sharex=True, sharey=True)
                ax = ax.flatten()

                for i in range(self.num_energy_bins):
                    bin_mask = (self.labels > self.ebins[i]) & (self.labels <= self.ebins[i + 1])
                    if np.sum(bin_mask) == 0:
                        continue
                    vox_metric1_bin = vox_metric1[bin_mask]
                    vox_metric2_bin = vox_metric2[bin_mask]
                    num_events_per_energy_bin = len(vox_metric1_bin)
                    frac_events_per_energy_bin = (num_events_per_energy_bin/len(vox_metric1))*100
                    
                    hist_counts_vox_metric1, bin_edges_vox_metric1 = np.histogram(vox_metric1_bin, bins=rbins, density=False)
                    hist_counts_vox_metric2, bin_edges_vox_metric2 = np.histogram(vox_metric2_bin, bins=rbins, density=False)
                    ax[i].hist(bin_edges_vox_metric1[:-1], bins=bin_edges_vox_metric1, weights=hist_counts_vox_metric1/num_events_per_energy_bin, label=f'{vox_metric_name1} [MeV]', alpha=0.5, edgecolor="none")
                    ax[i].hist(bin_edges_vox_metric2[:-1], bins=bin_edges_vox_metric2, weights=hist_counts_vox_metric2/num_events_per_energy_bin, label=f'{vox_metric_name2} [MeV]', alpha=0.5, edgecolor="none")
                    ax[i].set_xlabel("Average Voxel Energy [MeV]", fontsize=16)
                    ax[i].set_ylabel(f'Fraction of Test Events / {round(bin_width, 2)} [MeV]', fontsize=16)
                    ax[i].legend(loc='upper right', fontsize=16)
                    ax[i].tick_params(axis='both', which='major', labelsize=14)
                    ax[i].text(0.63, 0.8, f"{self.ebins[i]:.0f}-{self.ebins[i+1]:.0f} MeV \n{frac_events_per_energy_bin:.2f}% of Events", transform=ax[i].transAxes, fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', color='orange', alpha=0.2))
                    ax[i].legend(loc='upper right', fontsize=16)
                    self.num_events_by_ebin.append(num_events_per_energy_bin)

                fig.suptitle(f'Average Voxel Energy by True Energy Bin for {self.version}', size=20)
                fig.tight_layout()
                output.savefig(fig)
                self.num_events_by_ebin = np.array(self.num_events_by_ebin)
                plt.close()
            plot_avg_voxel_metrics_by_energy_bin()
            def plot_single_voxel_metric_by_ve_frac_bin(vox_metric=self.minE, vox_metric_name="Minimum Voxel Energy"):

                self.ve_frac_bins = np.linspace(0, 1, self.num_ve_frac_bins + 1)
                if vox_metric_name == 'Minimum Voxel Energy':
                    rbins = np.linspace(self.minE.min(), self.minE.max(), 1001)
                    if self.minE.min() == self.minE.max():
                        rbins = np.linspace(self.minE.min(), self.minE.max()+1, 1001)
                elif vox_metric_name == 'Maximum Voxel Energy':
                    rbins = np.linspace(self.maxE.min(), self.maxE.max(), int(round((self.maxE.max() - self.maxE.min()), 1)*10) + 1)
                elif vox_metric_name == 'Number of Filled Voxels':
                    rbins = np.linspace(self.numVox.min(), self.numVox.max(), int(round((self.numVox.max() - self.numVox.min()), 1)/100) + 1)
                else: 
                    print(f"Vox metric name of {vox_metric_name} unknown ... function is breaking.")
                    return
                bin_width = rbins[1] - rbins[0]
                self.num_events_by_ve_frac_bin = []

                fig, ax = plt.subplots(int(self.num_ve_frac_bins/5),5, figsize=(30, int(6*(self.num_ve_frac_bins/5))))#, sharex=True, sharey=True)
                ax = ax.flatten()

                for i in range(self.num_ve_frac_bins):
                    bin_mask = (self.ve / self.labels > self.ve_frac_bins[i]) & (self.ve / self.labels <= self.ve_frac_bins[i + 1])
                    if np.sum(bin_mask) == 0:
                        continue
                    vox_metric_bin = vox_metric[bin_mask]
                    num_events_per_ve_frac_bin = len(vox_metric_bin)
                    frac_events_per_ve_frac_bin = (num_events_per_ve_frac_bin/len(vox_metric))*100
                    
                    hist_counts_vox_metric, bin_edges_vox_metric = np.histogram(vox_metric_bin, bins=rbins, density=False)
                    ax[i].hist(bin_edges_vox_metric[:-1], bins=bin_edges_vox_metric, weights=hist_counts_vox_metric/num_events_per_ve_frac_bin, label=f'{vox_metric_name} [MeV]', alpha=0.5, edgecolor="none")
                    ax[i].set_xlabel(f'{vox_metric_name} [MeV]', fontsize=16)
                    ax[i].set_ylabel(f'Fraction of Test Events / {round(bin_width, 2)} [MeV]', fontsize=16)
                    ax[i].legend(loc='upper right', fontsize=16)
                    ax[i].tick_params(axis='both', which='major', labelsize=14)
                    ax[i].text(0.63, 0.8, f"{self.ve_frac_bins[i]:.2f}-{self.ve_frac_bins[i+1]:.2f} \n{frac_events_per_ve_frac_bin:.2f}% of Events", transform=ax[i].transAxes, fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', color='orange', alpha=0.2))
                    ax[i].legend(loc='upper right', fontsize=16)
                    self.num_events_by_ve_frac_bin.append(num_events_per_ve_frac_bin)

                fig.suptitle(f'{vox_metric_name} by Visible Energy Fraction Bin for {self.version}', size=20)
                fig.tight_layout()
                output.savefig(fig)
                self.num_events_by_ve_frac_bin = np.array(self.num_events_by_ve_frac_bin)
                plt.close()
            plot_single_voxel_metric_by_ve_frac_bin(self.minE, "Minimum Voxel Energy")
            plot_single_voxel_metric_by_ve_frac_bin(self.maxE, "Maximum Voxel Energy")
            plot_single_voxel_metric_by_ve_frac_bin(self.numVox, "Number of Filled Voxels")
            def plot_avg_voxel_metrics_by_ve_frac_bin(vox_metric1=self.meanE, vox_metric_name1="Mean Voxel Energy", vox_metric2=self.medE, vox_metric_name2="Median Voxel Energy"):

                self.ve_frac_bins = np.linspace(0, 1, self.num_ve_frac_bins + 1)
                vox_metric_min = min(vox_metric1.min(), vox_metric2.min())
                vox_metric_max = min(vox_metric1.max(), vox_metric2.max())
                rbins = np.linspace(vox_metric_min, vox_metric_max, int(round((vox_metric_max - vox_metric_min), 1) * 100) + 1)
                bin_width = rbins[1] - rbins[0]
                self.num_events_by_ve_frac_bin = []

                fig, ax = plt.subplots(int(self.num_ve_frac_bins/5),5, figsize=(30, int(6*(self.num_ve_frac_bins/5))))#, sharex=True, sharey=True)
                ax = ax.flatten()

                for i in range(self.num_ve_frac_bins):
                    bin_mask = (self.ve / self.labels > self.ve_frac_bins[i]) & (self.ve / self.labels <= self.ve_frac_bins[i + 1])
                    if np.sum(bin_mask) == 0:
                        continue
                    vox_metric1_bin = vox_metric1[bin_mask]
                    vox_metric2_bin = vox_metric2[bin_mask]
                    num_events_per_ve_frac_bin = len(vox_metric1_bin)
                    frac_events_per_ve_frac_bin = (num_events_per_ve_frac_bin/len(vox_metric1))*100
                    
                    hist_counts_vox_metric1, bin_edges_vox_metric1 = np.histogram(vox_metric1_bin, bins=rbins, density=False)
                    hist_counts_vox_metric2, bin_edges_vox_metric2 = np.histogram(vox_metric2_bin, bins=rbins, density=False)
                    ax[i].hist(bin_edges_vox_metric1[:-1], bins=bin_edges_vox_metric1, weights=hist_counts_vox_metric1/num_events_per_ve_frac_bin, label=f'{vox_metric_name1} [MeV]', alpha=0.5, edgecolor="none")
                    ax[i].hist(bin_edges_vox_metric2[:-1], bins=bin_edges_vox_metric2, weights=hist_counts_vox_metric2/num_events_per_ve_frac_bin, label=f'{vox_metric_name2} [MeV]', alpha=0.5, edgecolor="none")
                    ax[i].set_xlabel("Average Voxel Energy [MeV]", fontsize=16)
                    ax[i].set_ylabel(f'Fraction of Test Events / {round(bin_width, 2)} [MeV]', fontsize=16)
                    ax[i].legend(loc='upper right', fontsize=16)
                    ax[i].tick_params(axis='both', which='major', labelsize=14)
                    ax[i].text(0.63, 0.8, f"{self.ve_frac_bins[i]:.2f}-{self.ve_frac_bins[i+1]:.2f} \n{frac_events_per_ve_frac_bin:.2f}% of Events", transform=ax[i].transAxes, fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', color='orange', alpha=0.2))
                    ax[i].legend(loc='upper right', fontsize=16)
                    self.num_events_by_ve_frac_bin.append(num_events_per_ve_frac_bin)

                fig.suptitle(f'Average Voxel Energy by Visible Energy Fraction Bin for {self.version}', size=20)
                fig.tight_layout()
                output.savefig(fig)
                self.num_events_by_ve_frac_bin = np.array(self.num_events_by_ve_frac_bin)
                plt.close()
            plot_avg_voxel_metrics_by_ve_frac_bin()

            # Make start position plots
            print("Starting start position plots...")


            start = self.start_position
            x = start[:, 0]
            y = start[:, 1]
            z = start[:, 2]
            xbins = np.linspace(-370, 370, 149)
            ybins = np.linspace(-240, 105, 70)
            zbins = np.linspace(395, 935, 109)
            bin_width = xbins[1] - xbins[0]

            # 1) 2D projections
            fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
            h0 = axes[0].hist2d(x, y, bins=(xbins, ybins), cmap='magma')
            axes[0].set_xlabel("X (Drift) [cm]")
            axes[0].set_ylabel("Y (Vertical) [cm]")
            axes[0].set_title("Start Position: X-Y/Drift-Vertical")
            fig.colorbar(h0[3], ax=axes[0], label='Events')

            h1 = axes[1].hist2d(z, x, bins=(zbins, xbins), cmap='magma')
            axes[1].set_xlabel("Z (Beam) [cm]")
            axes[1].set_ylabel("X (Drift) [cm]")
            axes[1].set_title("Start Position: Z-X/Beam-Drift")
            fig.colorbar(h1[3], ax=axes[1], label='Events')

            h2 = axes[2].hist2d(z, y, bins=(zbins, ybins), cmap='magma')
            axes[2].set_xlabel("Z (Beam) [cm]")
            axes[2].set_ylabel("Y (Vertical) [cm]")
            axes[2].set_title("Start Position: Z-Y/Beam-Vertical")
            fig.colorbar(h2[3], ax=axes[2], label='Events')

            output.savefig(fig)
            plt.close(fig)

            # Overall 1D histograms
            fig, axes = plt.subplots(1, 3, figsize=(18, 4), constrained_layout=True)
            hist_counts_x, bin_edges_x = np.histogram(x, bins=xbins, density=False)
            hist_counts_y, bin_edges_y = np.histogram(y, bins=ybins, density=False)
            hist_counts_z, bin_edges_z = np.histogram(z, bins=zbins, density=False)
            axes[0].hist(bin_edges_x[:-1], bins=bin_edges_x, weights=hist_counts_x/num_events, histtype='stepfilled', alpha=0.7, color='steelblue')
            axes[0].set_xlabel('X/Drift [cm]')
            axes[0].set_ylabel(f'Fraction of Test Events / {round(bin_width, 2)} [cm]', fontsize=16)
            axes[0].set_title('Start Position: X/Drift')

            axes[1].hist(bin_edges_y[:-1], bins=bin_edges_y, weights=hist_counts_y/num_events, histtype='stepfilled', alpha=0.7, color='darkorange')
            axes[1].set_xlabel('Y/Vertical [cm]')
            axes[1].set_ylabel(f'Fraction of Test Events / {round(bin_width, 2)} [cm]', fontsize=16)
            axes[1].set_title('Start Position: Y/Vertical')

            axes[2].hist(bin_edges_z[:-1], bins=bin_edges_z, weights=hist_counts_z/num_events, histtype='stepfilled', alpha=0.7, color='seagreen')
            axes[2].set_xlabel('Z/Beam [cm]')
            axes[2].set_ylabel(f'Fraction of Test Events / {round(bin_width, 2)} [cm]', fontsize=16)
            axes[2].set_title('Start Position: Z/Beam')

            output.savefig(fig)
            plt.close(fig)

            # 1D histograms by energy bin
            def plot_start_pos_by_energy_bin(points1d, points1d_bins, points1d_name):
    
                self.num_events_by_ebin = []
                bin_width = points1d_bins[1] - points1d_bins[0]

                fig, ax = plt.subplots(int(self.num_energy_bins/5),5, figsize=(30, int(6*(self.num_energy_bins/5))))#, sharex=True, sharey=True)
                ax = ax.flatten()

                for i in range(self.num_energy_bins):
                    bin_mask = (self.labels > self.ebins[i]) & (self.labels <= self.ebins[i + 1])
                    if np.sum(bin_mask) == 0:
                        continue
                    points_in_bin = points1d[bin_mask]
                    num_events_per_energy_bin = len(points_in_bin)
                    frac_events_per_energy_bin = (num_events_per_energy_bin/len(points1d))*100
                    
                    hist_counts_pt, bin_edges_pt = np.histogram(points_in_bin, bins=points1d_bins, density=False)
                    ax[i].hist(bin_edges_pt[:-1], bins=bin_edges_pt, weights=hist_counts_pt/num_events_per_energy_bin, label=f'{points1d_name} [cm]', alpha=0.5, edgecolor="none")
                    ax[i].set_xlabel(f'{points1d_name} [cm]', fontsize=16)
                    ax[i].set_ylabel(f'Fraction of Events / {round(bin_width, 2)} [cm]', fontsize=16)
                    ax[i].legend(loc='upper right', fontsize=16)
                    ax[i].tick_params(axis='both', which='major', labelsize=14)
                    ax[i].text(0.63, 0.8, f"{self.ebins[i]:.0f}-{self.ebins[i+1]:.0f} MeV \n{frac_events_per_energy_bin:.2f}% of Events", transform=ax[i].transAxes, fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', color='orange', alpha=0.2))
                    ax[i].legend(loc='upper right', fontsize=16)
                    self.num_events_by_ebin.append(num_events_per_energy_bin)

                fig.suptitle(f'{points1d_name} by True Energy Bin for {self.version}', size=20)
                fig.tight_layout()
                output.savefig(fig)
                self.num_events_by_ebin = np.array(self.num_events_by_ebin)
                plt.close()
            plot_start_pos_by_energy_bin(x, xbins, "X/Drift")
            plot_start_pos_by_energy_bin(y, ybins, "Y/Vertical")
            plot_start_pos_by_energy_bin(z, zbins, "Z/Beam")

            # 1D histograms by visible energy fraction bin
            def plot_start_pos_by_ve_frac_bin(points1d, points1d_bins,points1d_name):

                self.num_events_by_ve_frac_bin = []
                bin_width = points1d_bins[1] - points1d_bins[0]

                fig, ax = plt.subplots(int(self.num_ve_frac_bins/5),5, figsize=(30, int(6*(self.num_ve_frac_bins/5))))#, sharex=True, sharey=True)
                ax = ax.flatten()

                for i in range(self.num_ve_frac_bins):
                    bin_mask = (self.ve / self.labels > self.ve_frac_bins[i]) & (self.ve / self.labels <= self.ve_frac_bins[i + 1])
                    if np.sum(bin_mask) == 0:
                        continue
                    points_in_bin = points1d[bin_mask]
                    num_events_per_ve_frac_bin = len(points_in_bin)
                    frac_events_per_ve_frac_bin = (num_events_per_ve_frac_bin/len(points1d))*100
                    
                    hist_counts_pt, bin_edges_pt = np.histogram(points_in_bin, bins=points1d_bins, density=False)
                    ax[i].hist(bin_edges_pt[:-1], bins=bin_edges_pt, weights=hist_counts_pt/num_events_per_ve_frac_bin, label=f'{points1d_name} [cm]', alpha=0.5, edgecolor="none")
                    ax[i].set_xlabel(f'{points1d_name} [cm]', fontsize=16)
                    ax[i].set_ylabel(f'Fraction of Events / {round(bin_width, 2)} [cm]', fontsize=16)
                    ax[i].legend(loc='upper right', fontsize=16)
                    ax[i].tick_params(axis='both', which='major', labelsize=14)
                    ax[i].text(0.63, 0.8, f"{self.ve_frac_bins[i]:.2f}-{self.ve_frac_bins[i+1]:.2f} \n{frac_events_per_ve_frac_bin:.2f}% of Events", transform=ax[i].transAxes, fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', color='orange', alpha=0.2))
                    ax[i].legend(loc='upper right', fontsize=16)
                    self.num_events_by_ve_frac_bin.append(num_events_per_ve_frac_bin)

                fig.suptitle(f'{points1d_name} by Visible Energy Fraction Bin for {self.version}', size=20)
                fig.tight_layout()
                output.savefig(fig)
                self.num_events_by_ve_frac_bin = np.array(self.num_events_by_ve_frac_bin)
                plt.close()
            plot_start_pos_by_ve_frac_bin(x, xbins, "X/Drift")
            plot_start_pos_by_ve_frac_bin(y, ybins, "Y/Vertical")
            plot_start_pos_by_ve_frac_bin(z, zbins, "Z/Beam")'''

    def run(self):
        self.get_values_from_test_csv()
        self.make_plots()

       

if __name__ == '__main__':
    # parsers for any cmd line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config", default='./configs/plot_sheep.yaml', type=str)
    args = parser.parse_args()
    params = ParseYAML(os.path.abspath(args.yaml_config), "default")

    sheep_test = TestedSheepAll(params)
    sheep_test.run()
