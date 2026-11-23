import os
import csv
import torch
import numpy as np
from datetime import datetime



def save_array_to_csv(data, file_path, header=None):
    'Save a scalar, vector, tensor, or matrix as CSV.'
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    data = np.array(data)

    try:
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if header is not None:
                writer.writerow(header)

            
            if data.ndim == 0:  
                writer.writerow([data.item()])
            elif data.ndim == 1:  
                writer.writerow(data)
            else:  
                writer.writerows(data)
    except Exception as e:
        print(f'[ERROR] Failed to save CSV {file_path}:{str(e)}')


# ==========================================

# ==========================================
class ExperimentUtils:
    @staticmethod
    def set_global_seed(seed):
        'Set global seed.'
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    @staticmethod
    def setup_experiment_directories(base_dir, n_data):
        'Setup experiment directories.'
        experiment_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        root_dir = os.path.join(base_dir, f"Exp_SLCP_{experiment_time}")
        n_data_dir = os.path.join(root_dir, f"n_data_{n_data}")
        os.makedirs(n_data_dir, exist_ok=True)
        return root_dir, n_data_dir

    @staticmethod
    def save_dict_to_csv_fixed(data_dict, file_path):
        'Save dict to csv fixed.'
        try:
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['metric_name', 'metric_value'])
                for key, value in data_dict.items():
                    
                    if isinstance(value, (list, np.ndarray, torch.Tensor)):
                        value = str(value)
                    writer.writerow([key, value])
        except Exception as e:
            print(f'[ERROR] Failed to save metric file: {e}')

    @staticmethod
    def save_summaries(root_dir, params_summary=None, metrics_summary=None, mle_comparison_summary=None):
        'Save summaries.'
        
        if params_summary and len(params_summary) > 0:
            p_path = os.path.join(root_dir, "all_final_parameters_summary.csv")
            with open(p_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=params_summary[0].keys())
                writer.writeheader()
                writer.writerows(params_summary)
            print(f'[OK] Global parameter summary saved: {p_path}')

        
        if metrics_summary and len(metrics_summary) > 0:
            m_path = os.path.join(root_dir, "all_evaluation_metrics_summary.csv")
            with open(m_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=metrics_summary[0].keys())
                writer.writeheader()
                writer.writerows(metrics_summary)
            print(f'[OK] Global metric summary saved: {m_path}')

        
        if mle_comparison_summary and len(mle_comparison_summary) > 0:
            c_path = os.path.join(root_dir, "mle_comparison_summary.csv")
            with open(c_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=mle_comparison_summary[0].keys())
                writer.writeheader()
                writer.writerows(mle_comparison_summary)
            print(f'[OK] MLE-comparison summary saved: {c_path}')
