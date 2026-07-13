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
from scipy.stats import linregress, skew, kurtosis, norm, crystalball
from scipy.optimize import curve_fit, minimize, Bounds, differential_evolution
from mpl_toolkits.mplot3d.axes3d import Axes


def gaussian(x, amplitude, mean, std_dev):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * std_dev ** 2))

def binned_cb_nll(params, bin_centers, counts, bin_width, num_events, reflect=False):
    beta, m, loc, scale = params
    if beta <= 0 or m <= 1 or scale <= 0:
        return np.inf
    x = -bin_centers if reflect else bin_centers
    counts = counts / num_events  # Normalize counts to fraction of events
    expected = bin_width * crystalball.pdf(x, beta, m, loc, scale)
    # Only use bins with expected > 0 to avoid log(0)
    mask = expected > 0
    nll = -np.sum(counts[mask] * np.log(expected[mask]) - expected[mask])
    return nll

def binned_gaussian_nll(params, bin_centers, counts, bin_width, num_events):
    loc, scale = params
    if scale <= 0:
        return np.inf
    counts = counts / num_events  # Normalize counts to fraction of events
    #mask = counts > 0  # Only consider bins with observed counts > 0
    mask = counts > 0
    expected = bin_width * norm.pdf(bin_centers[mask], loc, scale)
    valid = expected > 0
    nll = -np.sum(counts[mask][valid] * np.log(expected[valid]) - expected[valid])
    return nll

from scipy.integrate import quad

def double_cb_pdf_unnorm(x, beta_l, m_l, beta_r, m_r, loc, scale):
    """Unnormalized double-sided crystal ball."""
    x_shifted = (x - loc)
    return np.where(
        x_shifted <= 0,
        crystalball.pdf(x_shifted, beta_l, m_l, 0, scale),  # left tail
        crystalball.pdf(-x_shifted, beta_r, m_r, 0, scale),  # right tail
    )

def double_cb_pdf(x, beta_l, m_l, beta_r, m_r, loc, scale):
    """Normalized double-sided crystal ball PDF."""
    norm_factor, _ = quad(double_cb_pdf_unnorm, -np.inf, np.inf,
                          args=(beta_l, m_l, beta_r, m_r, loc, scale))
    return double_cb_pdf_unnorm(x, beta_l, m_l, beta_r, m_r, loc, scale) / (norm_factor)

def binned_double_cb_nll(params, bin_centers, counts, bin_width, num_events):
    beta_l, m_l, beta_r, m_r, loc, scale = params
    if beta_l <= 0 or m_l <= 1 or beta_r <= 0 or m_r <= 1 or scale <= 0:
        return np.inf
    counts = counts / num_events  # Normalize counts to fraction of events
    mask = counts > 0
    # Continuity penalty
    val_l = crystalball.pdf(loc, beta_l, m_l, loc, scale)
    val_r = crystalball.pdf(loc, beta_r, m_r, loc, scale)
    penalty = 10 * (val_l - val_r)**2  # large weight forces continuity
    try:
        expected = bin_width * double_cb_pdf(bin_centers[mask], beta_l, m_l, beta_r, m_r, loc, scale)
    except Exception:
        return np.inf
    valid = expected > 0
    nll = -np.sum(counts[mask][valid] * np.log(expected[valid]) - expected[valid])
    return nll+penalty



class TestedSheepNDLAr():

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
        self.output_pdf_name = '{}_{}_{}_NUEtest_results.pdf'.format(self.run_num, self.config, self.checkpoint_file.split('.')[0])
        os.makedirs(self.output_pdf_dir, exist_ok=True)

    def get_values_from_csv(self):

        labels = []
        preds = []
        ve = []
        ve_frac = []
        mg_frac = []
        oob_frac = []
        start_position = []
        rotation_matrix = []
        with open(self.csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                #print(row)
                labels.append(float(row['label']))
                preds.append(float(row['prediction']))
                ve.append(float(row['visible_energy']))        
                ve_frac.append(float(row['ve_frac']))
                mg_frac.append(float(row['mg_frac']))
                oob_frac.append(float(row['oob_frac']))
                start_position.append(str(row['start_position']))
                rotation_matrix.append(str(row['rotation_matrix']))

        self.labels = np.array(labels)
        self.preds = np.array(preds)
        self.ve = np.array(ve)
        self.ve_frac = np.array(ve_frac)
        self.mg_frac = np.array(mg_frac)
        self.oob_frac = np.array(oob_frac)
        self.start_position = np.array([np.fromstring(s.strip('[]'), sep=' ') for s in start_position])
        self.rotation_matrix = np.array([
            np.fromstring(r.replace('\n', ' ').replace('[', '').replace(']', ''), sep=' ').reshape(3, 3)
                            for r in rotation_matrix])

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
            y_eq_x = np.linspace(0,15010, 15010)
            ax.plot(y_eq_x, y_eq_x, 'k' '--', alpha=0.7, label="Perfect Prediction")
            ax.set_xlim(-5,15010)
            ax.set_ylim(-5,16000)
            res = linregress(self.labels, self.preds)
            fitted_line = res.slope * y_eq_x + res.intercept
            ax.plot(y_eq_x, fitted_line, color='rebeccapurple', label='Linear Best Fit', alpha=0.9, linewidth=2)
            ax.text(10466, 15472, f'Slope: {res.slope:.2f}\nIntercept: {res.intercept:.2f}\nR²: {res.rvalue**2:.2f}', fontsize=11, color='rebeccapurple', verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.2))
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

                y_eq_x = np.linspace(0, 15010, 15010)
                ax[i].plot(y_eq_x, y_eq_x, 'k' '--', alpha=0.7, linewidth=3, label='Perfect Prediction')
                ax[i].set_xlim(-5, 15010)
                ax[i].set_ylim(-5, 16000)
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
            self.ebins = np.linspace(0, 15000, self.num_energy_bins + 1)
            self.true_vs_pred_res_slope_by_ebin = []
            self.true_vs_pred_res_intercept_by_ebin = []
            self.true_vs_pred_res_r2_by_ebin = []
            self.num_events_by_ebin = []
            fig, ax = plt.subplots(6,5, figsize=(45, 24), sharey=True)
            ax = ax.flatten()
    
            for i in range(self.num_energy_bins):
                bin_mask = (self.labels > self.ebins[i]) & (self.labels <= self.ebins[i + 1])
                if np.sum(bin_mask) == 0:
                    continue
                ax[i].scatter(self.labels[bin_mask], self.preds[bin_mask], c=self.ve[bin_mask]/self.labels[bin_mask], cmap='Spectral', vmin=0, vmax=1., alpha=0.7, s=1, label=f"{self.ebins[i]:.0f}-{self.ebins[i+1]:.0f} MeV")

                y_eq_x = np.linspace(0, 15010, 15010)
                ax[i].plot(y_eq_x, y_eq_x, 'k' '--', alpha=0.7, linewidth=3, label='Perfect Prediction')
                ax[i].set_xlim(self.ebins[i]-2, self.ebins[i+1]+2)
                ax[i].set_ylim(-5, 16000)
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
            data_min = min(res_true_pred.min(), res_vis_true.min())
            data_max = max(res_true_pred.max(), res_vis_true.max())
            rbins = np.linspace(data_min, data_max, int(round((data_max - data_min), 2) * 50) + 1)
            bin_width = rbins[1] - rbins[0]
            num_events=len(res_vis_true)

            hist_counts_vis_true, bin_edges_vis_true = np.histogram(res_vis_true, bins=rbins, density=False)
            bin_centers_vis_true = (bin_edges_vis_true[:-1] + bin_edges_vis_true[1:]) / 2
            initial_guesses_vis_true  = [np.max(hist_counts_vis_true), np.mean(res_vis_true), np.std(res_vis_true)]

            hist_counts_true_pred, bin_edges_true_pred = np.histogram(res_true_pred, bins=rbins, density=False)
            bin_centers_true_pred = (bin_edges_true_pred[:-1] + bin_edges_true_pred[1:]) / 2
            initial_guesses_true_pred  = [np.max(hist_counts_true_pred / num_events), np.mean(res_true_pred), np.std(res_true_pred)]
            #print(f"Initial Guesses for (SHEEP - True) / True Gaussian Fit: {initial_guesses_true_pred}")
            q3, q1 = np.percentile(res_true_pred, [75, 25])
            iqr_value = q3 - q1

            print(f"IQR Value: {iqr_value}")
            # Perform the curve fit
            #vis_true_params, vis_true_covariance = curve_fit(gaussian, bin_centers_vis_true, hist_counts_vis_true, p0=initial_guesses_vis_true)
            #true_pred_params, true_pred_covariance = curve_fit(gaussian, bin_centers_true_pred, hist_counts_true_pred/ num_events, p0=initial_guesses_true_pred)
            bounds = Bounds(
                                lb=[0.1, 1.01, -np.inf, 0.001],
                                ub=[10.0, 50.0,  np.inf, np.inf]
                            )
            bounds_gauss = Bounds(
                lb=[-0.1, 0.001],
                ub=[0.1, np.inf]
            )

            bounds_dcb = Bounds(
                lb=[0.1, 1.01, 0.1, 1.01, -0.1, 0.001],
                ub=[1.0, 50.0, 1.0, 50.0,  0.1, 0.15]
            )
            bounds_list = list(zip(bounds_dcb.lb, bounds_dcb.ub))


            # --- Pred ---
            #x0_pred = [0.2, 3.0, 0.2, 3.0, 0, 0.03]
            #result_dcb_pred = differential_evolution(
            #    binned_double_cb_nll,
            #    bounds_list,
            #    args=(bin_centers_true_pred, hist_counts_true_pred, bin_width, num_events),
            #    maxiter=300,
            #    popsize=20,
            #    seed=42,
            #    polish=True
            #)
            #dcb_params_pred = result_dcb_pred.x
            #dcb_params_pred = result_dcb_pred.x
            seeds = [
                #[0.25, 25.0, 0.25, 25.0, 0.0, 0.05],
                #[0.5, 10.0, 0.5, 10.0, 0.0, 0.08],
                #[1.0, 5.0, 1.0, 5.0, 0.0, 0.10],
                #[0.2, 50.0, 0.2, 50.0, 0.0, 0.03],
                #[0.1, 2.0, 0.1, 2.0, 0.0, 0.01],
                [0.15, 5.0, 0.15, 5.0, 0.0, 0.02],
                #[0.3, 8.0, 0.3, 8.0, 0.0, 0.04],
                #[0.1, 30.0, 0.1, 30.0, 0.0, 0.025],
            ]
        
            best_result = None
            best_nll = np.inf
            
            for x0_pred in seeds:
                result = minimize(
                    binned_double_cb_nll,
                    x0=x0_pred,
                    args=(bin_centers_true_pred, hist_counts_true_pred, bin_width, num_events),
                    method='L-BFGS-B',
                    bounds=bounds_dcb
                )
                print("Result for seed {}: success={}, nll={:.4f}".format(x0_pred, result.success, result.fun))
                if result.success and result.fun < best_nll:
                    best_nll = result.fun
                    best_result = result
        
            dcb_params_pred = best_result.x
            print(f"DCB pred converged: {best_result.success} — {best_result.message}")
            print(f"beta_l={dcb_params_pred[0]:.3f}, m_l={dcb_params_pred[1]:.3f}, "
                  f"beta_r={dcb_params_pred[2]:.3f}, m_r={dcb_params_pred[3]:.3f}, "
                  f"loc={dcb_params_pred[4]:.3f}, scale={dcb_params_pred[5]:.3f}")

            # --- Vis (not reflected) ---
            x0_vis = [1.0, 2.0, np.mean(res_vis_true), np.std(res_vis_true)]
            result_vis = minimize(
                binned_cb_nll,
                x0=x0_vis,
                args=(bin_centers_vis_true, hist_counts_vis_true, bin_width, num_events, False),
                method='L-BFGS-B',
                bounds=bounds
            )
            cb_params_vis = result_vis.x
            print(f"Vis converged:  {result_vis.success} — {result_vis.message}")
            #x_cb_pred = np.linspace(res_true_pred.min(), res_true_pred.max(), 10000)
            #x_cb_vis = np.linspace(res_vis_true.min(), res_vis_true.max(), 10000)
            #self.pred_res_total_skew = skew(res_true_pred)
            #self.pred_res_total_kurtosis = kurtosis(res_true_pred)



            fig, (ax_main, ax_res) = plt.subplots(2, 1, figsize=(8, 8), 
                                        gridspec_kw={'height_ratios': [3, 1]}, 
                                        sharex=True)
            ax_main.hist(bin_edges_vis_true[:-1], bins=bin_edges_vis_true, weights=hist_counts_vis_true/ num_events, label="(Visible - True) / True", alpha=0.5, edgecolor="none")
            #plt.plot(bin_centers_vis_true, gaussian(bin_centers_vis_true, *vis_true_params), color='blue', linestyle="--")
            ax_main.hist(bin_edges_true_pred[:-1],bins=bin_edges_true_pred, weights=hist_counts_true_pred/ num_events, label="(SHEEP - True) / True", alpha=0.5, edgecolor="none")
            ax_main.plot(bin_centers_true_pred, bin_width * double_cb_pdf(bin_centers_true_pred, *dcb_params_pred), color='sienna', linestyle="--", linewidth=1)
            ax_main.plot(bin_centers_vis_true, bin_width * crystalball.pdf(bin_centers_vis_true, *cb_params_vis), color='blue', linestyle="--", linewidth=1)
            #ax.plot(bin_centers_true_pred, gaussian(bin_centers_true_pred, *true_pred_params), color='sienna', linestyle="--")
            ax_main.legend(fontsize=11)
            ax_main.set_xlim(-2, 2.3)
            #ax_main.set_ylim(0, 0.5)
            ax_main.set_ylabel(f"Fraction of Test Events / {bin_width:.2f}")
            ax_main.set_xlabel("Test Event Energy Resolution")
            ax=plt.gca()
            #ax_main.text(0.8, 0.23, r"$\mathbf{Mean:}$"+f"{cb_params_pred[2]:.2f}\n"+r"$\mathbf{Std Dev:}$"+f"{cb_params_pred[3]:.2f}\n"+r"$\mathbf{Skew:}$"+f"{self.pred_res_total_skew:.2f}\n"+r"$\mathbf{Kurtosis:}$"+f"{self.pred_res_total_kurtosis:.2f}", fontsize=12, verticalalignment='top', color='sienna')
            ax_main.text(0.8, 0.2, r"$\mathbf{Mean:}$"+f"{dcb_params_pred[4]:.2f}\n"+r"$\mathbf{Std Dev:}$"+f"{dcb_params_pred[5]:.2f}\n"+r"$\mathbf{\alpha_l:}$"+f"{dcb_params_pred[0]:.2f}\n"+r"$\mathbf{n_l:}$"+f"{dcb_params_pred[1]:.2f}\n"+r"$\mathbf{\alpha_r:}$"+f"{dcb_params_pred[2]:.2f}\n"+r"$\mathbf{n_r:}$"+f"{dcb_params_pred[3]:.2f}", fontsize=12, verticalalignment='top', color='sienna')
            ax_main.text(0.8, 0.1, r"$\mathbf{Mean:}$"+f"{cb_params_vis[2]:.2f}\n"+r"$\mathbf{Std Dev:}$"+f"{cb_params_vis[3]:.2f}\n"+r"$\mathbf{\alpha:}$"+f"{cb_params_vis[0]:.2f}\n"+r"$\mathbf{n:}$"+f"{cb_params_vis[1]:.2f}", fontsize=12, verticalalignment='top', color='blue')
            #ax_main.text(-1.88, 0.267, f"PRELIMINARY", fontsize=18, verticalalignment='top', color='black', alpha=0.35, fontweight='bold')
            ax_main.text(-1.88, 0.267, self.version, fontsize=12, verticalalignment='top', color='black', alpha=0.8, fontweight='bold')

            # --- Residual plot ---
            # Expected values from CB fits, evaluated at bin centers
            expected_pred = bin_width * double_cb_pdf(bin_centers_true_pred, *dcb_params_pred) # note: reflected
            expected_vis  = bin_width * crystalball.pdf(bin_centers_vis_true, *cb_params_vis)
            
            observed_pred = hist_counts_true_pred / num_events
            observed_vis  = hist_counts_vis_true / num_events
            
            # Poisson errors on the normalized counts
            errors_pred = np.sqrt(hist_counts_true_pred) / num_events
            errors_vis  = np.sqrt(hist_counts_vis_true)  / num_events
            errors_pred[errors_pred == 0] = (1 / num_events) / num_events  # avoid division by zero
            errors_vis[errors_vis == 0]   = (1 / num_events) / num_events
            
            pulls_pred = (observed_pred - expected_pred) / errors_pred
            pulls_vis  = (observed_vis  - expected_vis)  / errors_vis
            
            ax_res.bar(bin_centers_true_pred, pulls_pred, width=bin_width, alpha=0.5, color='sienna', label='SHEEP')
            ax_res.bar(bin_centers_vis_true,  pulls_vis,  width=bin_width, alpha=0.5, color='blue',   label='Visible')
            ax_res.axhline(0,  color='black', linewidth=0.8)
            ax_res.axhline(+2, color='gray',  linewidth=0.8, linestyle=':')
            ax_res.axhline(-2, color='gray',  linewidth=0.8, linestyle=':')
            ax_res.set_ylabel("Pull (σ)")
            ax_res.set_xlabel("Test Event Energy Resolution")
            ax_res.set_ylim(-5, 5)
            ax_res.set_xlim(-2, 2.3)

            # --- Chi2/NDF for pred (reflected CB) ---
            expected_pred = bin_width * double_cb_pdf(bin_centers_true_pred, *dcb_params_pred)
            observed_pred = hist_counts_true_pred / num_events

            # Only use bins with enough counts
            mask_pred = observed_pred > (5 / num_events)
            n_params = 6  # amplitude, loc, scale
            ndf_pred = mask_pred.sum() - n_params

            chi2_pred = np.sum((observed_pred[mask_pred] - expected_pred[mask_pred])**2 / expected_pred[mask_pred])
            print(f"Pred CB: chi2/ndf = {chi2_pred:.2f} / {ndf_pred} = {chi2_pred/ndf_pred:.2f}")

            # --- Chi2/NDF for vis CB ---
            expected_vis = bin_width * crystalball.pdf(bin_centers_vis_true, *cb_params_vis)
            observed_vis = hist_counts_vis_true / num_events

            mask_vis = observed_vis > (5 / num_events)
            ndf_vis = mask_vis.sum() - 4

            chi2_vis = np.sum((observed_vis[mask_vis] - expected_vis[mask_vis])**2 / expected_vis[mask_vis])
            print(f"Vis CB:  chi2/ndf = {chi2_vis:.2f} / {ndf_vis} = {chi2_vis/ndf_vis:.2f}")


            fig.tight_layout()
            output.savefig(fig)
            self.pred_total_res_gauss_fit_mean = dcb_params_pred[4]#true_pred_params[1]
            self.pred_total_res_gauss_fit_std = dcb_params_pred[5]#true_pred_params[2]
            plt.close()

            #### OOB Frac vs. MG Frac by visible energy fraction bin
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


            #### True - Predicted / True Energy (All Points) Fit to Gaussian by energy bin
            self.pred_res_gauss_fit_mean_by_ebin = []
            self.pred_res_gauss_fit_std_by_ebin = []
            self.pred_res_skew_by_ebin = []
            self.pred_res_kurtosis_by_ebin = []
            self.num_events_by_ebin = []
            ebin_centers = 0.5 * (self.ebins[:-1] + self.ebins[1:])
            data_min = min(res_true_pred.min(), res_vis_true.min())
            data_max = max(res_true_pred.max(), res_vis_true.max())
            rbins = np.linspace(data_min, data_max, int(round((data_max - data_min), 2) * 25) + 1)
            bin_width = rbins[1] - rbins[0]

            fig, ax = plt.subplots(int(self.num_energy_bins/5),5, figsize=(30, int(6*(self.num_energy_bins/5))))#, sharex=True, sharey=True)
            ax = ax.flatten()

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
                #data_min = min(res_true_pred.min(), res_vis_true.min())
                #data_max = max(res_true_pred.max(), res_vis_true.max())
                #rbins = np.linspace(data_min, data_max, int(round((data_max - data_min), 2) * 50) + 1)
                #bin_width = rbins[1] - rbins[0]

                hist_counts_vis_true, bin_edges_vis_true = np.histogram(res_vis_true, bins=rbins, density=False)
                bin_centers_vis_true = (bin_edges_vis_true[:-1] + bin_edges_vis_true[1:]) / 2
                initial_guesses_vis_true  = [np.max(hist_counts_vis_true/num_events_per_energy_bin), np.mean(res_vis_true), np.std(res_vis_true)]

                hist_counts_true_pred, bin_edges_true_pred = np.histogram(res_true_pred, bins=rbins, density=False)
                bin_centers_true_pred = (bin_edges_true_pred[:-1] + bin_edges_true_pred[1:]) / 2
                initial_guesses_true_pred  = [np.max(hist_counts_true_pred/num_events_per_energy_bin), np.mean(res_true_pred), np.std(res_true_pred)]
                #true_pred_params, true_pred_covariance = curve_fit(gaussian, bin_centers_true_pred, (hist_counts_true_pred/num_events_per_energy_bin), p0=initial_guesses_true_pred)
                pred_res_skew = skew(res_true_pred)
                pred_res_kurtosis = kurtosis(res_true_pred)

                            # --- Pred  ---
                x0_pred = [np.mean(res_true_pred), np.std(res_true_pred)]
                result_pred = minimize(
                    binned_gaussian_nll,
                    x0=x0_pred,
                    args=(bin_centers_true_pred, hist_counts_true_pred, bin_width, num_events_per_energy_bin),
                    method='L-BFGS-B',
                    bounds=bounds_gauss
                )
                fit_params_pred = result_pred.x
                print(f"Pred converged: {result_pred.success}")

                # --- Vis (not reflected) ---
                x0_vis = [1.0, 2.0, np.mean(res_vis_true), np.std(res_vis_true)]
                result_vis = minimize(
                    binned_cb_nll,
                    x0=x0_vis,
                    args=(bin_centers_vis_true, hist_counts_vis_true, bin_width, num_events_per_energy_bin, False),
                    method='L-BFGS-B',
                    bounds=bounds
                )
                cb_params_vis = result_vis.x
                print(f"Vis converged:  {result_vis.success} — {result_vis.message}")

                ax[i].hist(bin_edges_vis_true[:-1], bins=rbins, weights=hist_counts_vis_true/num_events_per_energy_bin, label=f'(Visible-True)/True', alpha=0.5)
                ax[i].hist(bin_edges_true_pred[:-1], bins=rbins, weights=hist_counts_true_pred/num_events_per_energy_bin, label=f'(SHEEP-True)/True', alpha=0.5)
                ax[i].plot(bin_centers_true_pred, bin_width * norm.pdf(bin_centers_true_pred, *fit_params_pred), color='sienna', linestyle="--", linewidth=1)
                ax[i].plot(bin_centers_vis_true, bin_width * crystalball.pdf(bin_centers_vis_true, *cb_params_vis), color='blue', linestyle="--", linewidth=1)
                ax[i].set_xlim(-2, 3)
                ax[i].set_ylim(-0, 0.28)
                ax[i].set_xlabel('Event Energy Resolution', fontsize=16)
                ax[i].set_ylabel(f'Fraction of Test Events / {round(bin_width, 2)}', fontsize=16)
                ax[i].legend(loc='upper right', fontsize=16)
                ax[i].tick_params(axis='both', which='major', labelsize=14)
                ax[i].text(0.63, 0.8, f"{self.ebins[i]:.0f}-{self.ebins[i+1]:.0f} MeV \n{frac_events_per_energy_bin:.2f}% of Events\nMean: {fit_params_pred[0]:.2f} \nStd Dev: {fit_params_pred[1]:.2f}", transform=ax[i].transAxes, fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', color='orange', alpha=0.2))
                ax[i].text(0.63, 0.4, f"Mean: {cb_params_vis[2]:.2f} \nStd Dev: {cb_params_vis[3]:.2f}\nα: {cb_params_vis[0]:.2f}\nn: {cb_params_vis[1]:.2f}", transform=ax[i].transAxes, fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', color='blue', alpha=0.2))
                ax[i].legend(loc='upper right', fontsize=16)
                self.pred_res_gauss_fit_mean_by_ebin.append(fit_params_pred[0])
                self.pred_res_gauss_fit_std_by_ebin.append(fit_params_pred[1])
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
            rbins = np.linspace(-10, 10, 501)

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
    parser.add_argument("--num_energy_bins", default=30, type=int, help='number of energy bins for plots')
    parser.add_argument("--num_ve_frac_bins", default=10, type=int, help='number of visible energy fraction bins for plots')
    args = parser.parse_args()

    sheep_test = TestedSheepNDLAr(args)
