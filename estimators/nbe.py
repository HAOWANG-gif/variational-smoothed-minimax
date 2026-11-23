import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import copy
import time
import warnings
import json
import glob

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _training_metrics_path(model_path):
    return f"{model_path}.training_metrics.json"


def _save_training_metrics(model_path, training_time, total_virtual_samples):
    metrics_path = _training_metrics_path(model_path)
    tmp_path = f"{metrics_path}.tmp.{os.getpid()}"
    payload = {
        "training_time": float(training_time),
        "total_virtual_samples": int(total_virtual_samples),
    }

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    os.replace(tmp_path, metrics_path)


def apply_nbe_training_cost_once(records, root_dir):
    'Apply nbe training cost once.'
    nbe_records = [
        record for record in records
        if str(record.get("method", "")).strip() == "NBE"
    ]

    if not nbe_records:
        return records

    def _positive(value):
        try:
            return float(value) > 0
        except (TypeError, ValueError):
            return False

    
    
    if any(_positive(record.get("n_sim")) for record in nbe_records):
        return records

    search_root = os.path.dirname(os.path.abspath(root_dir))
    pattern = os.path.join(
        search_root,
        "**",
        "NBE_piecewise_model.pth.training_metrics.json",
    )
    metrics_paths = glob.glob(pattern, recursive=True)

    if not metrics_paths:
        return records

    metrics_path = max(metrics_paths, key=os.path.getmtime)
    with open(metrics_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    training_time = float(payload.get("training_time", 0.0))
    total_virtual_samples = int(payload.get("total_virtual_samples", 0))

    if training_time <= 0 and total_virtual_samples <= 0:
        return records

    nbe_records[0]["time_used"] = training_time
    nbe_records[0]["n_sim"] = total_virtual_samples
    return records


def _xavier_init(module):
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
class PreSimulatedDataset(Dataset):
    def __init__(self, Z, theta):
        self.Z = Z
        self.theta = theta

    def __len__(self):
        return self.Z.shape[0]

    def __getitem__(self, idx):
        return (torch.from_numpy(self.Z[idx]), torch.from_numpy(self.theta[idx]))

def subset_data(Z, m_subset):
    return Z[:, :m_subset, :]

class SummaryNetwork(nn.Module):
    def __init__(self, x_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 256), nn.ReLU(inplace=True)
        )
        self.apply(_xavier_init)

    def forward(self, x):
        return self.net(x)
class InferenceNetwork(nn.Module):
    def __init__(self, theta_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(256, 500), nn.ReLU(inplace=True),
            nn.Linear(500, theta_dim),
        )
        self.apply(_xavier_init)

    def forward(self, x):
        return self.net(x)
class DeepSet(nn.Module):
    def __init__(self, x_dim, theta_dim):
        super().__init__()
        self.psi = SummaryNetwork(x_dim)
        self.phi = InferenceNetwork(theta_dim)

    def forward(self, x):
        batch_size, m, features = x.shape
        x_flat = x.reshape(batch_size * m, features)
        psi_out = self.psi(x_flat)
        psi_out = psi_out.view(batch_size, m, 256)
        aggregated = torch.mean(psi_out, dim=1)
        estimates = self.phi(aggregated)
        return estimates


class PiecewiseEstimator(nn.Module):
    def __init__(self, m_values, m_cutoffs, x_dim, theta_dim):
        super().__init__()
        self.m_values = list(m_values)
        self.m_cutoffs = list(m_cutoffs)
        self.x_dim = x_dim
        self.theta_dim = theta_dim
        self.networks = nn.ModuleList([DeepSet(x_dim, theta_dim) for _ in range(len(m_values))])

    def get_network_index(self, m):
        for i, cutoff in enumerate(self.m_cutoffs):
            if m <= cutoff: return i
        return len(self.networks) - 1

    def forward(self, x, m):
        idx = self.get_network_index(m)
        return self.networks[idx](x)

    def get_network(self, idx):
        return self.networks[idx]


class CosineAnnealingLR:
    def __init__(self, optimizer, max_lr, period):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.period = period
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        progress = self.current_epoch / self.period
        lr = self.max_lr * (1.0 + np.cos(np.pi * progress)) / 2.0
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


class AmortizedTrainer:
    def __init__(self, piecewise_estimator, sample_generator_cls, train_config):
        self.estimator = piecewise_estimator
        self.sample_generator_cls = sample_generator_cls

        
        self.m_values = train_config.get("m_values", [1, 10, 50, 100])
        self.m_cutoffs = train_config.get("m_cutoffs", [5, 30, 75])
        self.K_train = train_config.get("K_train", 100000)
        self.K_val = train_config.get("K_val", 2000)
        self.batch_size = train_config.get("batch_size", 32)
        self.epochs = train_config.get("epochs", 300)
        self.x_dim = train_config.get("x_dim", 8)
        self.theta_dim = train_config.get("theta_dim", 5)
        self.patience=train_config.get("patience", 4)
        self.patience0=train_config.get("patience0", 10)
        
        self.target_transform = train_config.get("target_transform", None)

        self.LOSS = train_config.get("LOSS", "L1")
        if self.LOSS == "L1":
            self.criterion = nn.L1Loss()
        elif self.LOSS == "MSE":
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported LOSS: {self.LOSS}")
        self.patience_map = {1: self.patience0}
        for m in self.m_values[1:]:
            self.patience_map[m] = self.patience


        self.theta_train = None
        self.theta_val = None
        self.Z_train_np = None
        self.Z_val_np = None
        self.history = []
        self.learn1=train_config.get("learn1", 1e-4)
        self.learn2 = train_config.get("learn2", 1e-5)
    def _generate_data(self, K, m):
        theta = self.sample_generator_cls.sample_prior(n_samples=K).cpu()
        Z = torch.zeros((K, m, self.x_dim))
        for i in range(K):
            gen = self.sample_generator_cls(theta[i], n_samples=m)
            Z[i] = gen.generate_samples().cpu().view(m, self.x_dim)
        return theta.numpy(), Z.numpy()

    def setup(self):
        m_train = 2 * max(self.m_values)
        self.theta_train, self.Z_train_np = self._generate_data(self.K_train, m_train)

        m_val = max(self.m_values)
        self.theta_val, self.Z_val_np = self._generate_data(self.K_val, m_val)

    def train_stage(self, stage_idx):
        m_i = self.m_values[stage_idx]
        network = self.estimator.get_network(stage_idx)
        network.to(device)
        network.train()

        lr = self.learn1 if m_i == 1 else self.learn2
        optimizer = optim.Adam(network.parameters(), lr=lr)
        scheduler = CosineAnnealingLR(optimizer, lr, self.epochs)
        patience = self.patience_map[m_i]

        train_dataset = PreSimulatedDataset(self.Z_train_np, self.theta_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        Z_val_sub = subset_data(self.Z_val_np, m_i)
        val_dataset = PreSimulatedDataset(Z_val_sub, self.theta_val)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        best_val_loss = float('inf')
        best_state = None
        no_improve = 0

        for epoch in range(self.epochs):
            network.train()
            train_loss_sum = 0.0
            train_samples = 0

            for data_batch, theta_batch in train_loader:
                total_m = data_batch.size(1)
                rand_indices = torch.randperm(total_m)[:m_i]
                data_batch = data_batch[:, rand_indices, :].to(device)
                theta_batch = theta_batch.to(device)
                batch_size_actual = data_batch.size(0)

                # ==========================================
                
                # ==========================================
                if self.target_transform is not None:
                    theta_target = self.target_transform(theta_batch)
                else:
                    theta_target = theta_batch

                optimizer.zero_grad()
                pred = network(data_batch)

                loss = self.criterion(pred, theta_target)
                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item() * batch_size_actual
                train_samples += batch_size_actual

            avg_train_loss = train_loss_sum / train_samples if train_samples > 0 else float('inf')

            network.eval()
            val_loss_sum = 0.0
            val_samples = 0
            with torch.no_grad():
                for data_batch, theta_batch in val_loader:
                    data_batch = data_batch.to(device)
                    theta_batch = theta_batch.to(device)
                    batch_size_actual = data_batch.size(0)

                    
                    if self.target_transform is not None:
                        theta_target = self.target_transform(theta_batch)
                    else:
                        theta_target = theta_batch

                    pred = network(data_batch)
                    loss = self.criterion(pred, theta_target)
                    val_loss_sum += loss.item() * batch_size_actual
                    val_samples += batch_size_actual

            avg_val_loss = val_loss_sum / val_samples if val_samples > 0 else float('inf')
            scheduler.step()

            self.history.append({'epoch': epoch + 1, 'train_loss': avg_train_loss, 'val_loss': avg_val_loss})

            if avg_val_loss <= best_val_loss:
                best_val_loss = avg_val_loss
                best_state = copy.deepcopy(network.state_dict())
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                break

        if best_state is not None:
            network.load_state_dict(best_state)

        return best_val_loss

    def train_all(self, resume_from=0):
        self.setup()
        for stage_idx in range(resume_from, len(self.m_values)):
            if stage_idx > 0:
                prev_state = copy.deepcopy(self.estimator.get_network(stage_idx - 1).state_dict())
                self.estimator.get_network(stage_idx).load_state_dict(prev_state)
            self.train_stage(stage_idx)

    def save(self, path='nbe_model.pth'):
        
        state_dicts = []
        for net in self.estimator.networks:
            cpu_net = copy.deepcopy(net).cpu()
            state_dicts.append(cpu_net.state_dict())

        torch.save({
            'm_values': self.m_values,
            'm_cutoffs': self.m_cutoffs,
            'x_dim': self.x_dim,
            'theta_dim': self.theta_dim,
            'networks_state_dict': state_dicts,
            'history': self.history,
        }, path)

    def load(self, path='nbe_model.pth'):
        load_dev = torch.device("cpu") if not torch.cuda.is_available() else device
        ckpt = torch.load(path, map_location=load_dev, weights_only=False)
        self.m_values = ckpt['m_values']
        self.m_cutoffs = ckpt['m_cutoffs']
        self.x_dim = ckpt.get('x_dim', 8)
        self.theta_dim = ckpt.get('theta_dim', 5)
        for i, sd in enumerate(ckpt['networks_state_dict']):
            self.estimator.networks[i].load_state_dict(sd)
        self.history = ckpt.get('history', [])


# ===================================================================

# ===================================================================

class AmortizedNBEEstimator:
    def __init__(self, method_name, n_data, sample_generator_cls, train_config):
        self.method_name = method_name
        self.n_data = n_data
        self.sample_generator_cls = sample_generator_cls
        self.train_config = train_config

        self.x_dim = train_config.get("x_dim", 8)
        self.theta_dim = train_config.get("theta_dim", 5)

        self.model_dir = train_config.get("model_dir", "./saved_nbe_models")
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_path = os.path.join(
            self.model_dir,
            f"{method_name}_piecewise_model.pth"
        )

        self.m_values = train_config.get("m_values", [1, 10, 50, 100])
        self.m_cutoffs = train_config.get("m_cutoffs", [5, 30, 75])

        
        self.total_virtual_samples = 0

    def fit(self, y_obs):
        start_time = time.time()

        estimator = PiecewiseEstimator(
            self.m_values,
            self.m_cutoffs,
            self.x_dim,
            self.theta_dim
        )

        trainer = AmortizedTrainer(
            estimator,
            self.sample_generator_cls,
            self.train_config
        )

        if os.path.exists(self.model_path):
            print(f'    --> Found a pretrained model; loading {self.model_path}')

            trainer.load(self.model_path)

            train_time = 0.0

            
            self.total_virtual_samples = 0

        else:
            train_start = time.time()

            print(f'    --> No usable {self.method_name} pretrained model; generating data and training...')

            trainer.train_all()
            trainer.save(self.model_path)

            train_time = time.time() - train_start

            # ==================================================
            
            
            
            
            # ==================================================
            K_train = self.train_config.get("K_train", 100000)
            K_val = self.train_config.get("K_val", 2000)

            m_train = 2 * max(self.m_values)
            m_val = max(self.m_values)

            self.total_virtual_samples = (
                K_train * m_train
                + K_val * m_val
            )
            _save_training_metrics(
                model_path=self.model_path,
                training_time=train_time,
                total_virtual_samples=self.total_virtual_samples,
            )
            print(f'    --> Training completed and the NBE model was saved; elapsed: {train_time:.2f}s')
            print(f'    --> NBE offline-training simulated-sample count: {self.total_virtual_samples}')

        eval_start = time.time()

        network_idx = estimator.get_network_index(self.n_data)
        network = estimator.get_network(network_idx)

        network.to(device)
        network.eval()

        with torch.no_grad():
            y_input = y_obs.view(1, self.n_data, self.x_dim).to(device)
            pred = network(y_input)
            map_estimate = pred.squeeze(0)

        total_time = train_time + (time.time() - eval_start)

        return (
            map_estimate,
            trainer.history,
            [],
            total_time,
            self.total_virtual_samples
        )
