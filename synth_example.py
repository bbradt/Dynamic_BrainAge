import os
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from catalyst import dl, utils
from dataloaders.synth_data import SynthTCs
from models.bilstm import BiLSTM
from sklearn.model_selection import train_test_split
import numpy as np
import torch

from saliency import get_saliency_synth





cuda = torch.device('cuda')


criterion = torch.nn.CrossEntropyLoss()
full_dataset = SynthTCs()
model = BiLSTM(seqlen=full_dataset.seqlen, dim=full_dataset.dim)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
full_idx = np.arange(len(full_dataset))
train_idx, test_idx = train_test_split(full_idx, test_size=0.1)
train_dataset = Subset(full_dataset, train_idx)
test_dataset = Subset(full_dataset, test_idx)

loaders = {
    "train": DataLoader(train_dataset, batch_size=32),
    "valid": DataLoader(test_dataset, batch_size=len(test_dataset)),
}

runner = dl.SupervisedRunner(
    input_key="features", output_key="logits", target_key="targets", loss_key="loss"
)

# model training
runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    loaders=loaders,
    num_epochs=10,
    callbacks=[
        dl.CriterionCallback(metric_key="loss",
                             input_key="logits",
                             target_key="targets"),
    ],
    logdir="./logs/test_2000_v2",
    valid_loader="valid",
    valid_metric="loss",
    minimize_valid_metric=True,
    verbose=True,
)


test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
checkpoint = utils.load_checkpoint("logs/test_2000_v2/checkpoints/best_full.pth")
utils.unpack_checkpoint(
    checkpoint=checkpoint,
    model=model,
    optimizer=optimizer,
    criterion=criterion
)
full_dataset_new = SynthTCs(ret_starts=True, samples=2000)
full_loader = DataLoader(full_dataset_new, batch_size=1)

# for data, label in test_loader:
#     prediction = model(data.to(cuda))
#     label = label.to(cuda)
#     stacked_labels = torch.stack([prediction, label], 0).squeeze()
#     corr = torch.corrcoef(stacked_labels)
#     print("Correlation with Ground Truth from Bets Model ", corr)
    
sals, data, starts = get_saliency_synth(model, full_loader)
print(sals.shape)
np.save("sals", sals)
np.save("data",data)
np.save("starts",starts)