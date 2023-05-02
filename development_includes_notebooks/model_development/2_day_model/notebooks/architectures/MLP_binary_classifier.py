import lightning.pytorch as pl
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from torch import nn
import torch
from torch import Tensor

from typing import Union,Optional
from typing import List

import torchmetrics as tm

class MLP(nn.Module):
    def __init__(
        self,
        embed_dims: Union[int, list],
        num_layers: Optional[int] = None,
        activation: str = "relu",
    ) -> None:
        super().__init__()

        if isinstance(embed_dims, int):
            assert num_layers is not None
            embed_dims = [embed_dims] * num_layers
        else:
            assert num_layers is None
            num_layers = len(embed_dims)

        assert activation == "relu"

        self.embed_dims = embed_dims
        self.num_layers = num_layers
        self.activation = activation

        self.layers = nn.ModuleList(
            [
                sublayer
                for i, (m, n) in enumerate(zip(embed_dims, embed_dims[1:]))
                for sublayer in [nn.Linear(m, n), nn.BatchNorm1d(n), nn.ReLU()]
                if not (i == num_layers - 2 and isinstance(sublayer, nn.ReLU))
            ]
        )


    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class MLP_binary_classifier(pl.LightningModule):
    
    def __init__(self,
                 embed_dims: List[int],
                 lr = 0.001):
        super().__init__()
        self.mlp = MLP(embed_dims)
        self.Classifier_head = nn.Linear(embed_dims[-1],1)
        self.s = nn.Sigmoid()
        self.step_outputs = {"train": [],"val":[],"test":[]}
        self.average_precision = tm.AveragePrecision(task = 'binary')
        self.auroc = {name : tm.AUROC(task = 'binary') for name in ['train','val']}
        self.lr = lr
        self.pr_curve = tm.PrecisionRecallCurve(task="binary")
        self.save_hyperparameters()
        
    def forward(self,
                batch):
        
        x,y = batch
        x = self.mlp(x)
        x = self.Classifier_head(x)
        x = self.s(x)
        
        return x
          
    def _step(self,
              name: str,
              batch: List[torch.tensor]):
        
        target = batch[1]#.to(device)
        predicted_proba = self.forward(batch).flatten()
        criterion = nn.BCELoss()
        loss = criterion(predicted_proba,target)
        self.step_outputs[name].append({"loss": loss, "predicted_proba": predicted_proba, "target": batch[1]})
        
        self.log(
            f"{name}/loss", loss, prog_bar = True, on_epoch=True, logger = True, on_step= (name=="train"), batch_size=batch[1].shape[0]
        )
        
        if name == 'val':
            x = target.type(torch.int)
            self.average_precision(predicted_proba,x)
            self.log("val/average_precision",self.average_precision,on_epoch = True,on_step = False)
        
        
        self.pr_curve.update(predicted_proba,target.type(torch.int))
        
        return {"loss": loss, "predicted_proba": predicted_proba, "target": batch[1]}
    
    def training_step(self, batch, batch_idx = None):
        return self._step("train", batch)
    
    def validation_step(self, batch, batch_idx = None):
        return self._step("val", batch)
   
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer  
    
    
    def on_validation_epoch_end(self):
        outputs = self.step_outputs['val']
        precision,recall,thresh = self.pr_curve.compute()
        
        a1 = precision[recall >= 0.25][-1]
        a2 = precision[recall >= 0.5][-1]
        a3 = precision[recall >= 0.75][-1]
        
        self.log("val/p@r=0.25",a1)
        self.log("val/p@r=0.5",a2)
        self.log("val/p@r=0.75",a3)
        
        self.pr_curve.reset()
        