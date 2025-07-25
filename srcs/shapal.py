import torch
import numpy as np
import pandas as pd
import shap
from scipy.stats import zscore
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, Subset
from data import DataLoader as DL
from Regressor import MultiOutputRegressor3l as MR3

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 24,
    'axes.titlesize': 28,
    'axes.labelsize': 26,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'legend.fontsize': 24,
    'axes.unicode_minus': False,
    'axes.linewidth': 2,
    'xtick.major.size': 8,
    'xtick.major.width': 2,
    'ytick.major.size': 8,
    'ytick.major.width': 2,
})


shap_colors = [
    '#a8d1ff', '#b7e1cd', '#fcfca6', '#ffd6ba', '#ffbad2', '#d1c4e9'  
]
background_color = '#f8f9fa' 


def z_score_normalize(data):
    normalized_data = zscore(data)
    return normalized_data


class SHAPModelWrapper:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    def __call__(self, x):
        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            outputs = self.model(x_tensor)
        return outputs.cpu().numpy()


def visualize_shap_results(shap_values, X_test, feature_names):
    n_features = X_test.shape[1]
    n_samples = X_test.shape[0]

    if len(feature_names) != n_features:
        feature_names = [f'Feature_{i + 1}' for i in range(n_features)]

    if len(shap_values) == n_samples:
        shap_values_per_output = []
        for out_idx in range(12):
            shap_for_output = np.zeros((n_samples, n_features))
            for i in range(n_samples):
                shap_for_output[i] = shap_values[i][:, out_idx]
            shap_values_per_output.append(shap_for_output)
        shap_values = shap_values_per_output

    n_outputs = len(shap_values)
    output_names = [
        r'$I_l$', r'$\lambda_l$', r'$I_t$', r'$\lambda_t$',
        r'$I_{\text{CD}+}$', r'$\lambda_{\text{CD}+}$',
        r'$I_{\text{CD}-}$', r'$\lambda_{\text{CD}-}$',
        r'$I_{g+}$', r'$\lambda_{g+}$',
        r'$I_{g-}$', r'$\lambda_{g-}$',
    ]
    plt.figure(figsize=(12, 9))
    overall_importance = np.zeros(n_features)
    for i in range(n_outputs):
        overall_importance += np.abs(shap_values[i]).mean(axis=0)
    overall_importance /= n_outputs

    bars = plt.barh(feature_names, overall_importance, color=shap_colors[:n_features])
    plt.xlabel('Mean Absolute SHAP Value', fontsize=26)
    plt.title('Average Feature Importance Across All Outputs', fontsize=28, pad=20)
    plt.gca().set_facecolor(background_color)
    plt.tight_layout()
    plt.savefig('overall_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

    for i in range(n_outputs):
        if shap_values[i].shape[1] != n_features:
            continue

        plt.figure(figsize=(12, 9))

        shap.summary_plot(
            shap_values[i],
            X_test,
            feature_names=feature_names,
            show=False,
            plot_type='dot',
            color_bar=True 
        )
        plt.gca().set_facecolor(background_color)
        plt.gca().tick_params(axis='both', labelsize=20)

        plt.xlabel(f'SHAP Value (impact on output {output_names[i]})', fontsize=20)

        cbar = plt.gcf().axes[-1]  
        cbar.tick_params(labelsize=22)  
        cbar.set_ylabel(cbar.get_ylabel(), fontsize=22)  

        plt.tight_layout(rect=[0, 0, 0.9, 1])
        plt.savefig(f'output_{i + 1}_shap_summary.png', dpi=300, bbox_inches='tight')
        plt.close()

    for i in range(n_outputs):
        if shap_values[i].shape[1] != n_features:
            continue

        for j, feat_name in enumerate(feature_names):
            plt.figure(figsize=(12, 9))
            try:
                dep_fig = plt.gcf()
                shap.dependence_plot(
                    j,
                    shap_values[i],
                    X_test,
                    feature_names=feature_names,
                    show=False,
                    dot_size=50,
                    alpha=0.6
                )
                plt.gca().set_facecolor(background_color)
                plt.gca().tick_params(axis='both', labelsize=22)
                plt.xlabel(feat_name, fontsize=26)
                plt.ylabel('SHAP Value', fontsize=22)

                if len(dep_fig.axes) > 1: 
                    cbar = dep_fig.axes[-1]
                    cbar.tick_params(labelsize=24)

                plt.tight_layout()
                plt.savefig(f'dependence_{feat_name}_output_{i + 1}.png', dpi=300, bbox_inches='tight')
            except Exception as e:
                print(f"Skipping dependence plot for {feat_name}: {str(e)}")
            finally:
                plt.close()

    print(f"Generated {1 + n_outputs + n_outputs * n_features} plots for paper.")


def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {DEVICE}")
    dataset = DL('./dataall.CSV')
    data = dataset.load_data()
    x = data.iloc[:, :6]
    from scipy.stats import zscore
    x = z_score_normalize(x)
    y = z_score_normalize(y)

    x_data = torch.tensor(x.values, dtype=torch.float32)
    y_data = torch.tensor(y.values, dtype=torch.float32)

    indice = list(range(1633))
    test_idx = indice[1389:]
    dataset = TensorDataset(x_data, y_data)
    test_set = Subset(dataset, test_idx)
    test_loader = DataLoader(test_set, batch_size=500, shuffle=False, num_workers=0)
    model = MR3(input_dim=6, output_dim=12, hid1=25, hid2=50, hid3=25)
    weight = torch.load('model.pth', map_location=DEVICE)
    model.load_state_dict(weight)
    model.to(DEVICE)
    model.eval()

    test_features = []
    for X, _ in test_loader:
        test_features.append(X.cpu().numpy())
    test_features = np.vstack(test_features)
    print(f"Test set shape: {test_features.shape}")
    feature_names = ['AR', 'p', 'd', 'N', 'L', '2r']
    model_wrapper = SHAPModelWrapper(model, DEVICE)
    background_samples = test_features[:min(100, len(test_features))]
    explainer = shap.KernelExplainer(model_wrapper, background_samples)
    n_samples = min(200, len(test_features))
    print(f"Calculating SHAP values for {n_samples} samples...")
    shap_values = explainer.shap_values(test_features[:n_samples], nsamples=100)
    visualize_shap_results(shap_values, test_features[:n_samples], feature_names)
    print("SHAP analysis completed. Plots saved.")


if __name__ == "__main__":
    main()