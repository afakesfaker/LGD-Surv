# WGAN-GP
# GAN model to augument survival data

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import ks_2samp, wasserstein_distance

#Input data
df = pd.read_csv("data.csv")
df_death = df[df["vital_status"] == 1].copy()
X = df_death.drop(columns=["vital_status", "days_alive"]).values
y = df_death["days_alive"].values.reshape(-1, 1)

scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y)

#parameters
latent_dim = 64 #32, 128
batch_size = 64 #32, 128
num_epochs = 50000 #5000, 10000, 100000
lambda_gp = 10 #5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_dim = X_scaled.shape[1]
condition_dim = 0  

# Model
class JointGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.feature_out = nn.Linear(256, feature_dim)
        self.time_out = nn.Linear(256, 1)

    def forward(self, z):
        if condition_dim > 0:
            z = torch.cat([z, c], dim=1)
        shared = self.shared(z)
        return self.feature_out(shared), self.time_out(shared)


class JointDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim + 1 + condition_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.out = nn.Linear(128, 1)

    def forward(self, x, t, return_features=False):
        input = torch.cat([x, t], dim=1)
        features = self.net(input)
        out = self.out(features)
        if return_features:
            return out, features
        return out

G = JointGenerator().to(device)
D = JointDiscriminator().to(device)

opt_G = optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.9))
opt_D = optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.9))

#WGAN-GP loss
def gradient_penalty(D, real_data, fake_data):
    alpha = torch.rand(real_data.size(0), 1).to(device)
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates.requires_grad_(True)
    d_interpolates = D(interpolates[:, :-1], interpolates[:, -1:])
    gradients = torch.autograd.grad(
        outputs=d_interpolates, inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

# Training
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_scaled, dtype=torch.float32)
dataloader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    for real_x, real_t in dataloader:
        real_x, real_t = real_x.to(device), real_t.to(device)

        z = torch.randn(real_x.size(0), latent_dim).to(device)
        fake_x, fake_t = G(z)
        real_input = torch.cat([real_x, real_t], dim=1)
        fake_input = torch.cat([fake_x.detach(), fake_t.detach()], dim=1)

        d_real = D(real_x, real_t)
        d_fake = D(fake_x.detach(), fake_t.detach())
        gp = gradient_penalty(D, real_input, fake_input)
        d_loss = -d_real.mean() + d_fake.mean() + lambda_gp * gp

        opt_D.zero_grad()
        d_loss.backward()
        opt_D.step()

        if epoch % 5 == 0:
            z = torch.randn(real_x.size(0), latent_dim).to(device)
            fake_x, fake_t = G(z)

            # Feature Matching Loss
            _, real_feat = D(real_x, real_t, return_features=True)
            _, fake_feat = D(fake_x, fake_t, return_features=True)
            fm_loss = ((real_feat.mean(dim=0) - fake_feat.mean(dim=0)) ** 2).mean()

            g_loss = -D(fake_x, fake_t).mean() + 0.1 * fm_loss

            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch} | D_loss: {d_loss.item():.4f} | G_loss: {g_loss.item():.4f}")

def postprocess_generated_samples(fake_x, fake_t, scaler_y, real_days_alive, apply_offset=True):
    fake_days_alive = scaler_y.inverse_transform(fake_t.cpu().detach().numpy())

    if apply_offset:
        offset = np.mean(real_days_alive) - np.mean(fake_days_alive)
        fake_days_alive = fake_days_alive + offset

    fake_days_alive = np.clip(fake_days_alive, a_min=0, a_max=None)

    return fake_days_alive.squeeze(), fake_x

# generate new samples
z = torch.randn(50, latent_dim).to(device)
gen_x, gen_t = G(z)
real_days_alive = y
gen_t_corrected, gen_x = postprocess_generated_samples(gen_x, gen_t, scaler_y, real_days_alive)
gen_x = scaler_X.inverse_transform(gen_x.detach().cpu().numpy())

numeric_columns = df_death.drop(columns=["vital_status", "days_alive"]).columns.tolist()
num_samples = gen_x.shape[0]

if "clinical_data_WHO" in numeric_columns:
    who_idx = numeric_columns.index("clinical_data_WHO")
    gen_x[:, who_idx] = np.random.choice([2, 3], num_samples)

binary_columns = [
    "clinical_data_gender",
    "clinical_data_history_seizures",
    "clinical_data_histologic_diagnosis_Astrocytoma",
    "clinical_data_histologic_diagnosis_Oligoastrocytoma",
    "clinical_data_histologic_diagnosis_Oligodendroglioma",
    "idh1_data_IDH1_R132H"
]
for col in binary_columns:
    if col in numeric_columns:
        idx = numeric_columns.index(col)
        gen_x[:, idx] = (gen_x[:, idx] > 0.5).astype(int)

def apply_binary_constraints(data, columns, colnames):
    for group in columns:
        idx = [colnames.index(col) for col in group if col in colnames]
        if not idx:
            continue
        choices = np.random.choice(len(idx), size=data.shape[0])
        binary_data = np.zeros((data.shape[0], len(idx)))
        binary_data[np.arange(data.shape[0]), choices] = 1
        data[:, idx] = binary_data

binary_constraints = [
    ["clinical_data_tumor_site1_Posterior Fossa", "clinical_data_tumor_site1_Supratentorial"],
    ["clinical_data_tumor_site2_Cerebellum", "clinical_data_tumor_site2_Frontal Lobe",
     "clinical_data_tumor_site2_Not Otherwise Specified", "clinical_data_tumor_site2_Occipital Lobe",
     "clinical_data_tumor_site2_Parietal Lobe", "clinical_data_tumor_site2_Temporal Lobe"],
    ["clinical_data_laterality_Left", "clinical_data_laterality_Midline", "clinical_data_laterality_Right"],
    ["clinical_data_supratentorial_localization_Cerebral Cortex",
     "clinical_data_supratentorial_localization_Deep Gray (e.g.basal ganglia, thalamus)",
     "clinical_data_supratentorial_localization_Not listed in Medical Record",
     "clinical_data_supratentorial_localization_White Matter"]
]
apply_binary_constraints(gen_x, binary_constraints, numeric_columns)

real_days_alive = real_days_alive.flatten()
generated_days_alive = gen_t_corrected

synthetic = pd.DataFrame(gen_x, columns=df_death.drop(columns=["vital_status", "days_alive"]).columns)
synthetic["days_alive"] = gen_t_corrected
synthetic["vital_status"] = 1

synthetic.to_csv("synthetic_data.csv", index=False)
