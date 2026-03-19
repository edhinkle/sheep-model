import numpy as np
from collections import namedtuple, OrderedDict

import matplotlib
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
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.axes import Axes
from scipy.stats import linregress, skew, kurtosis
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d.axes3d import Axes


def gaussian(x, amplitude, mean, std_dev):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * std_dev ** 2))


class TestedSheep():

    def __init__(self, args):

        self.csv_file = args.csv_file
        self.version = args.version
        self.config = args.config
        self.train_config = args.train_config
        self.results_dir = args.results_dir
        self.run_num = args.run_num
        self.checkpoint_file = args.checkpoint_file
        self.num_energy_bins = args.num_energy_bins
        self.num_ve_frac_bins = args.num_ve_frac_bins

        self.output_pdf_dir = os.path.join(self.results_dir, self.config, self.run_num, 'plots')
        self.output_pdf_name = '{}_{}_{}_test_results.pdf'.format(self.run_num, self.config, self.checkpoint_file.split('.')[0])
        os.makedirs(self.output_pdf_dir, exist_ok=True)

    def get_values_from_csv(self):

        labels = []
        preds = []
        ve = []
        with open(self.csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                #print(row)
                labels.append(float(row['label']))
                preds.append(float(row['prediction']))
                ve.append(float(row['visible_energy']))

        self.labels = np.array(labels)
        self.preds = np.array(preds)
        self.ve = np.array(ve)

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
            
            #### True vs. Predicted Energy (All Points) with color by visible energy fraction
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(self.labels, self.preds, c=self.ve/self.labels, cmap='Spectral', vmin=0, alpha=0.7, s=1)
            ax.plot([self.labels.min(), self.labels.max()], [self.labels.min(), self.labels.max()], 'r--')
            ax.set_xlabel('True KE [MeV]')
            ax.set_ylabel('Predicted Energy [MeV]')
            ax.set_title(f'{self.version} True vs. Predicted Energy')
            plt.colorbar(ax.collections[0], label='Visible Energy Fraction')
            y_eq_x = np.linspace(0,2010, 2010)
            ax.plot(y_eq_x, y_eq_x, 'k' '--', alpha=0.7, label="Perfect Prediction")
            ax.set_xlim(-5,2010)
            ax.set_ylim(-5,3000)
            res = linregress(self.labels, self.preds)
            fitted_line = res.slope * y_eq_x + res.intercept
            ax.plot(y_eq_x, fitted_line, color='rebeccapurple', label='Linear Best Fit', alpha=0.9, linewidth=2)
            ax.text(1400, 2900, f'Slope: {res.slope:.2f}\nIntercept: {res.intercept:.2f}\nR²: {res.rvalue**2:.2f}', fontsize=11, color='rebeccapurple', verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.2))
            ax.legend()
            output.savefig(fig)
            self.true_vs_pred_res_slope = res.slope
            self.true_vs_pred_res_intercept = res.intercept
            self.true_vs_pred_res_r2 = res.rvalue**2
            plt.close()


            #### True vs. Predicted Energy by visible energy fraction bin
            self.ve_frac_bins = np.linspace(0, 1, self.num_ve_frac_bins + 1)
            self.true_vs_pred_res_slope_by_ve_frac_bin = []
            self.true_vs_pred_res_intercept_by_ve_frac_bin = []
            self.true_vs_pred_res_r2_by_ve_frac_bin = []
            self.num_events_by_ve_frac_bin = []
            ve_frac_bin_centers = 0.5 * (self.ve_frac_bins[:-1] + self.ve_frac_bins[1:])
            fig, ax = plt.subplots(2,5, figsize=(27, 12), sharex=True, sharey=True)
            ax = ax.flatten()
    
            for i in range(self.num_ve_frac_bins):
                bin_mask = (self.ve / self.labels > self.ve_frac_bins[i]) & (self.ve / self.labels <= self.ve_frac_bins[i + 1])
                if np.sum(bin_mask) == 0:
                    continue
                ax[i].scatter(self.labels[bin_mask], self.preds[bin_mask], c=self.ve[bin_mask]/self.labels[bin_mask], cmap='Spectral', vmin=0, vmax=1., alpha=0.7, s=1, label=f'VE Fraction: {self.ve_frac_bins[i]:.2f}-{self.ve_frac_bins[i+1]:.2f}')

                y_eq_x = np.linspace(0, 2010, 2010)
                ax[i].plot(y_eq_x, y_eq_x, 'k' '--', alpha=0.7, linewidth=3, label='Perfect Prediction')
                ax[i].set_xlim(-5, 2010)
                ax[i].set_ylim(-5, 3000)
                ax[i].set_xlabel('True KE [MeV]', fontsize=16)
                ax[i].set_ylabel('Predicted Energy [MeV]', fontsize=16)
                ax[i].legend(loc='upper right', fontsize=16)
                ax[i].tick_params(axis='both', which='major', labelsize=14)

                frac_events = round(len(self.preds[bin_mask])/len(self.preds), 4)*100
                self.num_events_by_ve_frac_bin.append(len(self.preds[bin_mask]))
                x = self.labels[bin_mask]
                y = self.preds[bin_mask]
                #print(x.shape, y.shape)
                res = linregress(x,y)
                fitted_line = res.slope * y_eq_x + res.intercept
                ax[i].plot(y_eq_x, fitted_line, color='rebeccapurple', label='Linear Best Fit', alpha=0.9, linewidth=3)
                ax[i].text(0.05, 0.7, f'{frac_events:.2f}% of Events\nSlope: {res.slope:.2f}\nIntercept: {res.intercept:.2f}\nR²: {res.rvalue**2:.2f}', transform=ax[i].transAxes, color='rebeccapurple', fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
                self.true_vs_pred_res_slope_by_ve_frac_bin.append(res.slope)
                self.true_vs_pred_res_intercept_by_ve_frac_bin.append(res.intercept)
                self.true_vs_pred_res_r2_by_ve_frac_bin.append(res.rvalue**2)
                ax[i].legend(loc='upper right', fontsize=16)

            fig.suptitle(f'{self.version} True vs. Predicted Energy by Visible Energy Fraction Bin', size=20)
            fig.tight_layout()
            output.savefig(fig)
            self.true_vs_pred_res_slope_by_ve_frac_bin = np.array(self.true_vs_pred_res_slope_by_ve_frac_bin)
            self.true_vs_pred_res_intercept_by_ve_frac_bin = np.array(self.true_vs_pred_res_intercept_by_ve_frac_bin)
            self.true_vs_pred_res_r2_by_ve_frac_bin = np.array(self.true_vs_pred_res_r2_by_ve_frac_bin)
            self.num_events_by_ve_frac_bin = np.array(self.num_events_by_ve_frac_bin)
            plt.close()


            #### True vs. Predicted Energy by energy bin
            self.ebins = np.linspace(0, 2000, self.num_energy_bins + 1)
            self.true_vs_pred_res_slope_by_ebin = []
            self.true_vs_pred_res_intercept_by_ebin = []
            self.true_vs_pred_res_r2_by_ebin = []
            self.num_events_by_ebin = []
            fig, ax = plt.subplots(4,5, figsize=(30, 24), sharey=True)
            ax = ax.flatten()
    
            for i in range(self.num_energy_bins):
                bin_mask = (self.labels > self.ebins[i]) & (self.labels <= self.ebins[i + 1])
                if np.sum(bin_mask) == 0:
                    continue
                ax[i].scatter(self.labels[bin_mask], self.preds[bin_mask], c=self.ve[bin_mask]/self.labels[bin_mask], cmap='Spectral', vmin=0, vmax=1., alpha=0.7, s=1, label=f"{self.ebins[i]:.0f}-{self.ebins[i+1]:.0f} MeV")

                y_eq_x = np.linspace(0, 2010, 2010)
                ax[i].plot(y_eq_x, y_eq_x, 'k' '--', alpha=0.7, linewidth=3, label='Perfect Prediction')
                ax[i].set_xlim(self.ebins[i]-2, self.ebins[i+1]+2)
                ax[i].set_ylim(-5, 3000)
                ax[i].set_xlabel('True KE [MeV]', fontsize=16)
                ax[i].set_ylabel('Predicted Energy [MeV]', fontsize=16)
                ax[i].legend(loc='upper right', fontsize=16)
                ax[i].tick_params(axis='both', which='major', labelsize=14)

                frac_events = round(len(self.preds[bin_mask])/len(self.preds), 4)*100
                self.num_events_by_ebin.append(len(self.preds[bin_mask]))
                x = self.labels[bin_mask]
                y = self.preds[bin_mask]
                #print(x.shape, y.shape)
                res = linregress(x,y)
                fitted_line = res.slope * y_eq_x + res.intercept
                ax[i].plot(y_eq_x, fitted_line, color='rebeccapurple', label='Linear Best Fit', alpha=0.9, linewidth=3)
                ax[i].text(0.05, 0.7, f'{frac_events:.2f}% of Events\nSlope: {res.slope:.2f}\nIntercept: {res.intercept:.2f}\nR²: {res.rvalue**2:.2f}', transform=ax[i].transAxes, color='rebeccapurple', fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
                self.true_vs_pred_res_slope_by_ebin.append(res.slope)
                self.true_vs_pred_res_intercept_by_ebin.append(res.intercept)
                self.true_vs_pred_res_r2_by_ebin.append(res.rvalue**2)
                ax[i].legend(loc='upper right', fontsize=16)

            fig.suptitle(f'{self.version} True vs. Predicted Energy by Energy Bin', size=20)
            fig.tight_layout()
            output.savefig(fig)
            self.true_vs_pred_res_slope_by_ebin = np.array(self.true_vs_pred_res_slope_by_ebin)
            self.true_vs_pred_res_intercept_by_ebin = np.array(self.true_vs_pred_res_intercept_by_ebin)
            self.true_vs_pred_res_r2_by_ebin = np.array(self.true_vs_pred_res_r2_by_ebin)
            self.num_events_by_ebin = np.array(self.num_events_by_ebin)
            plt.close()


            #### True - Predicted / True Energy (All Points) Fit to Gaussian
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

            res_vis_true = (self.ve - self.labels) / self.labels
            res_true_pred = (self.preds - self.labels)/ self.labels
            rbins = np.linspace(-10, 10, 201)
            num_events=len(res_vis_true)

            hist_counts_vis_true, bin_edges_vis_true = np.histogram(res_vis_true, bins=rbins, density=False)
            bin_centers_vis_true = (bin_edges_vis_true[:-1] + bin_edges_vis_true[1:]) / 2
            initial_guesses_vis_true  = [np.max(hist_counts_vis_true), np.mean(res_vis_true), np.std(res_vis_true)]

            hist_counts_true_pred, bin_edges_true_pred = np.histogram(res_true_pred, bins=rbins, density=False)
            bin_centers_true_pred = (bin_edges_true_pred[:-1] + bin_edges_true_pred[1:]) / 2
            initial_guesses_true_pred  = [np.max(hist_counts_true_pred/num_events), np.mean(res_true_pred), np.std(res_true_pred)]

            # Perform the curve fit
            vis_true_params, vis_true_covariance = curve_fit(gaussian, bin_centers_vis_true, hist_counts_vis_true, p0=initial_guesses_vis_true)
            true_pred_params, true_pred_covariance = curve_fit(gaussian, bin_centers_true_pred, hist_counts_true_pred/num_events, p0=initial_guesses_true_pred)
            self.pred_res_total_skew = skew(res_true_pred)
            self.pred_res_total_kurtosis = kurtosis(res_true_pred)

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.hist(bin_edges_vis_true[:-1], bins=bin_edges_vis_true, weights=hist_counts_vis_true/num_events, label="(Visible - True) / True", alpha=0.5, edgecolor="none")
            #plt.plot(bin_centers_vis_true, gaussian(bin_centers_vis_true, *vis_true_params), color='blue', linestyle="--")
            ax.hist(bin_edges_true_pred[:-1],bins=bin_edges_true_pred, weights=hist_counts_true_pred/num_events, label="(SHEEP - True) / True", alpha=0.5, edgecolor="none")
            ax.plot(bin_centers_true_pred, gaussian(bin_centers_true_pred, *true_pred_params), color='sienna', linestyle="--")
            ax.legend(fontsize=11)
            ax.set_xlim(-2, 2.3)
            ax.set_ylim(0, 0.28)
            ax.set_ylabel("Fraction of Test Events / 0.1")
            ax.set_xlabel("Test Event Energy Resolution")
            ax=plt.gca()
            ax.text(0.8, 0.23, r"$\mathbf{Mean:}$"+f"{true_pred_params[1]:.2f}\n"+r"$\mathbf{Std Dev:}$"+f"{true_pred_params[2]:.2f}\n"+r"$\mathbf{Skew:}$"+f"{self.pred_res_total_skew:.2f}\n"+r"$\mathbf{Kurtosis:}$"+f"{self.pred_res_total_kurtosis:.2f}", fontsize=12, verticalalignment='top', color='sienna')
            #ax.text(-1.88, 0.267, f"PRELIMINARY", fontsize=18, verticalalignment='top', color='black', alpha=0.35, fontweight='bold')
            ax.text(-1.88, 0.267, self.version, fontsize=12, verticalalignment='top', color='black', alpha=0.8, fontweight='bold')
            fig.tight_layout()
            output.savefig(fig)
            self.pred_total_res_gauss_fit_mean = true_pred_params[1]
            self.pred_total_res_gauss_fit_std = true_pred_params[2]
            plt.close()


            #### True - Predicted / True Energy (All Points) Fit to Gaussian by energy bin
            self.pred_res_gauss_fit_mean_by_ebin = []
            self.pred_res_gauss_fit_std_by_ebin = []
            self.pred_res_skew_by_ebin = []
            self.pred_res_kurtosis_by_ebin = []
            self.num_events_by_ebin = []
            ebin_centers = 0.5 * (self.ebins[:-1] + self.ebins[1:])

            fig, ax = plt.subplots(int(self.num_energy_bins/5),5, figsize=(30, int(6*(self.num_energy_bins/5))))#, sharex=True, sharey=True)
            ax = ax.flatten()
            rbins = np.linspace(-10, 10, 201)

            for i in range(self.num_energy_bins):
                bin_mask = (self.labels > self.ebins[i]) & (self.labels <= self.ebins[i + 1])
                if np.sum(bin_mask) == 0:
                    continue
                ve_bin = self.ve[bin_mask]
                true_bin = self.labels[bin_mask]
                pred_bin =  self.preds[bin_mask]
                res_vis_true = (ve_bin-true_bin) / true_bin
                res_true_pred = (pred_bin-true_bin)/ true_bin
                num_events_per_energy_bin = len(true_bin)
                frac_events_per_energy_bin = (num_events_per_energy_bin/len(self.labels))*100

                hist_counts_vis_true, bin_edges_vis_true = np.histogram(res_vis_true, bins=rbins, density=False)
                bin_centers_vis_true = (bin_edges_vis_true[:-1] + bin_edges_vis_true[1:]) / 2
                initial_guesses_vis_true  = [np.max(hist_counts_vis_true/num_events_per_energy_bin), np.mean(res_vis_true), np.std(res_vis_true)]

                hist_counts_true_pred, bin_edges_true_pred = np.histogram(res_true_pred, bins=rbins, density=False)
                bin_centers_true_pred = (bin_edges_true_pred[:-1] + bin_edges_true_pred[1:]) / 2
                initial_guesses_true_pred  = [np.max(hist_counts_true_pred/num_events_per_energy_bin), np.mean(res_true_pred), np.std(res_true_pred)]
                true_pred_params, true_pred_covariance = curve_fit(gaussian, bin_centers_true_pred, (hist_counts_true_pred/num_events_per_energy_bin), p0=initial_guesses_true_pred)
                pred_res_skew = skew(res_true_pred)
                pred_res_kurtosis = kurtosis(res_true_pred)

                ax[i].hist(bin_edges_vis_true[:-1], bins=rbins, weights=hist_counts_vis_true/num_events_per_energy_bin, label=f'(Visible-True)/True', alpha=0.5)
                ax[i].hist(bin_edges_true_pred[:-1], bins=rbins, weights=hist_counts_true_pred/num_events_per_energy_bin, label=f'(SHEEP-True)/True', alpha=0.5)
                ax[i].plot(bin_centers_true_pred, gaussian(bin_centers_true_pred, *true_pred_params), color='sienna', linestyle="--")
                ax[i].set_xlim(-2, 3)
                ax[i].set_ylim(-0, 0.28)
                ax[i].set_xlabel('Event Energy Resolution', fontsize=16)
                ax[i].set_ylabel('Fraction of Test Events / 0.1', fontsize=16)
                ax[i].legend(loc='upper right', fontsize=16)
                ax[i].tick_params(axis='both', which='major', labelsize=14)
                ax[i].text(0.63, 0.8, f"{self.ebins[i]:.0f}-{self.ebins[i+1]:.0f} MeV \n{frac_events_per_energy_bin:.2f}% of Events\nMean: {true_pred_params[1]:.2f} \nStd Dev: {true_pred_params[2]:.2f}\nSkew: {pred_res_skew:.2f}\nKurtosis: {pred_res_kurtosis:.2f}", transform=ax[i].transAxes, fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', color='orange', alpha=0.2))

                ax[i].legend(loc='upper right', fontsize=16)
                self.pred_res_gauss_fit_mean_by_ebin.append(true_pred_params[1])
                self.pred_res_gauss_fit_std_by_ebin.append(true_pred_params[2])
                self.pred_res_skew_by_ebin.append(pred_res_skew)
                self.pred_res_kurtosis_by_ebin.append(pred_res_kurtosis)
                self.num_events_by_ebin.append(num_events_per_energy_bin)

            fig.suptitle(f'Sheep CNN Predicted and Visible Energy Resolutions by True Energy Bin for {self.version}', size=20)
            fig.tight_layout()
            output.savefig(fig)
            self.pred_res_gauss_fit_mean_by_ebin = np.array(self.pred_res_gauss_fit_mean_by_ebin)
            self.pred_res_gauss_fit_std_by_ebin = np.array(self.pred_res_gauss_fit_std_by_ebin)
            self.pred_res_skew_by_ebin = np.array(self.pred_res_skew_by_ebin)
            self.pred_res_kurtosis_by_ebin = np.array(self.pred_res_kurtosis_by_ebin)
            self.num_events_by_ebin = np.array(self.num_events_by_ebin)
            plt.close()


            #### True - Predicted / True Energy (All Points) Fit to Gaussian by visible energy fraction bin
            self.pred_res_gauss_fit_mean_by_ve_frac_bin = []
            self.pred_res_gauss_fit_std_by_ve_frac_bin = []
            self.pred_res_skew_by_ve_frac_bin = []
            self.pred_res_kurtosis_by_ve_frac_bin = []
            self.num_events_by_ve_frac_bin = []

            fig, ax = plt.subplots(int(self.num_ve_frac_bins/5),5, figsize=(30, int(6*(self.num_ve_frac_bins/5))))#, sharex=True, sharey=True)
            ax = ax.flatten()
            rbins = np.linspace(-10, 10, 201)

            for i in range(self.num_ve_frac_bins):
                bin_mask = (self.ve / self.labels > self.ve_frac_bins[i]) & (self.ve / self.labels <= self.ve_frac_bins[i + 1])
                if np.sum(bin_mask) == 0:
                    continue
                ve_bin = self.ve[bin_mask]
                true_bin = self.labels[bin_mask]
                pred_bin =  self.preds[bin_mask]
                res_vis_true = (ve_bin-true_bin) / true_bin
                res_true_pred = (pred_bin-true_bin)/ true_bin
                num_events_per_ve_frac_bin = len(true_bin)
                frac_events_per_ve_frac_bin = (num_events_per_ve_frac_bin/len(self.labels))*100

                hist_counts_vis_true, bin_edges_vis_true = np.histogram(res_vis_true, bins=rbins, density=False)
                bin_centers_vis_true = (bin_edges_vis_true[:-1] + bin_edges_vis_true[1:]) / 2
                initial_guesses_vis_true  = [np.max(hist_counts_vis_true/num_events_per_ve_frac_bin), np.mean(res_vis_true), np.std(res_vis_true)]

                hist_counts_true_pred, bin_edges_true_pred = np.histogram(res_true_pred, bins=rbins, density=False)
                bin_centers_true_pred = (bin_edges_true_pred[:-1] + bin_edges_true_pred[1:]) / 2
                initial_guesses_true_pred  = [np.max(hist_counts_true_pred/num_events_per_ve_frac_bin), np.mean(res_true_pred), np.std(res_true_pred)]
                true_pred_params, true_pred_covariance = curve_fit(gaussian, bin_centers_true_pred, (hist_counts_true_pred/num_events_per_ve_frac_bin), p0=initial_guesses_true_pred)
                pred_res_skew = skew(res_true_pred)
                pred_res_kurtosis = kurtosis(res_true_pred)

                ax[i].hist(bin_edges_vis_true[:-1], bins=rbins, weights=hist_counts_vis_true/num_events_per_ve_frac_bin, label=f'(Visible-True)/True', alpha=0.5)
                ax[i].hist(bin_edges_true_pred[:-1], bins=rbins, weights=hist_counts_true_pred/num_events_per_ve_frac_bin, label=f'(SHEEP-True)/True', alpha=0.5)
                ax[i].plot(bin_centers_true_pred, gaussian(bin_centers_true_pred, *true_pred_params), color='sienna', linestyle="--")
                ax[i].set_xlim(-2, 3)
                ax[i].set_ylim(-0, 0.28)
                ax[i].set_xlabel('Event Energy Resolution', fontsize=16)
                ax[i].set_ylabel('Fraction of Test Events / 0.1', fontsize=16)
                ax[i].legend(loc='upper right', fontsize=16)
                ax[i].tick_params(axis='both', which='major', labelsize=14)
                ax[i].text(0.63, 0.8, f"{self.ve_frac_bins[i]:.2f}-{self.ve_frac_bins[i+1]:.2f} \n{frac_events_per_ve_frac_bin:.2f}% of Events\nMean: {true_pred_params[1]:.2f} \nStd Dev: {true_pred_params[2]:.2f}\nSkew: {pred_res_skew:.2f}\nKurtosis: {pred_res_kurtosis:.2f}", transform=ax[i].transAxes, fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', color='orange', alpha=0.2))

                ax[i].legend(loc='upper right', fontsize=16)
                self.pred_res_gauss_fit_mean_by_ve_frac_bin.append(true_pred_params[1])
                self.pred_res_gauss_fit_std_by_ve_frac_bin.append(true_pred_params[2])
                self.pred_res_skew_by_ve_frac_bin.append(pred_res_skew)
                self.pred_res_kurtosis_by_ve_frac_bin.append(pred_res_kurtosis)
                self.num_events_by_ve_frac_bin.append(num_events_per_ve_frac_bin)

            fig.suptitle(f'Sheep CNN Predicted and Visible Energy Resolutions by Visible Energy Fraction Bin for {self.version}', size=20)
            fig.tight_layout()
            output.savefig(fig)
            self.pred_res_gauss_fit_mean_by_ve_frac_bin = np.array(self.pred_res_gauss_fit_mean_by_ve_frac_bin)
            self.pred_res_gauss_fit_std_by_ve_frac_bin = np.array(self.pred_res_gauss_fit_std_by_ve_frac_bin)
            self.pred_res_skew_by_ve_frac_bin = np.array(self.pred_res_skew_by_ve_frac_bin)
            self.pred_res_kurtosis_by_ve_frac_bin = np.array(self.pred_res_kurtosis_by_ve_frac_bin)
            self.num_events_by_ve_frac_bin = np.array(self.num_events_by_ve_frac_bin)
            plt.close()


    def run(self):
        self.get_values_from_csv()
        self.make_plots()

       

if __name__ == '__main__':
    # parsers for any cmd line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", default='test_results.csv', type=str, help='csv file with test results')
    parser.add_argument("--version", default='MSE Loss / E/1000 Target', type=str, help='csv file with test results')
    parser.add_argument("--config", default='test', type=str)
    parser.add_argument("--train_config", default='default', type=str)
    parser.add_argument("--results_dir", default='./outputs', type=str, help='directory to store results')
    parser.add_argument("--run_num", default='0', type=str, help='sub run config')
    parser.add_argument("--checkpoint_file", default='ckpt_best.tar', type=str, help='checkpoint file to load')
    parser.add_argument("--num_energy_bins", default=20, type=int, help='number of energy bins for plots')
    parser.add_argument("--num_ve_frac_bins", default=10, type=int, help='number of visible energy fraction bins for plots')
    args = parser.parse_args()

    sheep_test = TestedSheep(args)
