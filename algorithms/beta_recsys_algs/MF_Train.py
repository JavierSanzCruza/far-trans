#  Copyright (c) 2022. Terrier Team at University of Glasgow, http://http://terrierteam.dcs.gla.ac.uk
#
#  This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
#  If a copy of the MPL was not distributed with this  file, you can obtain one at
#  http://mozilla.org/MPL/2.0/.
#
#  This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
#  If a copy of the MPL was not distributed with this  file, you can obtain one at
#  http://mozilla.org/MPL/2.0/.

import os
import sys

import pandas as pd

sys.path.append("../")

import numpy as np
import torch

from beta_rec.core.train_engine import TrainEngine
from beta_rec.models.mf import MFEngine
from beta_rec.utils.monitor import Monitor

from beta_rec.utils.constants import (
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_USER_COL,
)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


class MF_train(TrainEngine):
    """An instance class from the TrainEngine base class."""

    def __init__(self, config, data, dataset_name, backup):
        """Initialize LightGCN_train Class.

        Args:
            config (dict): All the parameters for the model.
        """
        self.config = config
        super(MF_train, self).__init__(config)
        self.data = data.split
        self.dataset_name = dataset_name
        self.engine = None
        self.backup = backup

    def init(self):
        self.config["model"]["n_users"] = self.data.n_users
        self.config["model"]["n_items"] = self.data.n_items
        self.engine = MFEngine(self.config)

    def train(self):
        """Train the model."""
        #self.load_dataset()
        self.gpu_id, self.config["device_str"] = self.get_device()
        """ Main training navigator

        Returns:

        """

        self.model_save_dir = os.path.join(
            self.backup, self.config["model"]["save_name"]
        )
        # Train NeuMF without pre-train
        self.monitor = Monitor(
            log_dir=self.config["system"]["run_dir"], delay=1, gpu_id=self.gpu_id
        )
        if self.config["model"]["loss"] == "bpr":
            train_loader = self.data.instance_bpr_loader(
                batch_size=self.config["model"]["batch_size"],
                device=self.config["model"]["device_str"],
            )
        elif self.config["model"]["loss"] == "bce":
            train_loader = self.data.instance_bce_loader(
                num_negative=self.config["model"]["num_negative"],
                batch_size=self.config["model"]["batch_size"],
                device=self.config["model"]["device_str"],
            )
        else:
            raise ValueError(
                f"Unsupported loss type {self.config['loss']}, try other options: 'bpr' or 'bce'"
            )
        self.config["dataset"]["dataset"] = self.dataset_name
        self.config["dataset"]["data_split"] = "temporal"

        if os.path.exists(self.model_save_dir):
            return

        self.engine = MFEngine(self.config)
        self._train(self.engine, train_loader, self.model_save_dir)
        self.config["run_time"] = self.monitor.stop()
        return self.eval_engine.best_valid_performance

    def test(self):
        model_save_dir = os.path.join(
            self.backup, self.config["model"]["save_name"]
        )

        self.config["model"]["device_str"] = "cpu"
        self.engine = MFEngine(self.config)
        model = self.engine.resume_checkpoint(model_save_dir)

        aux_cust = [x for x in self.data.id2user.keys()]
        aux_assets = [x for x in self.data.id2item.keys()]

        index = pd.MultiIndex.from_product([aux_cust, aux_assets], names=[DEFAULT_USER_COL, DEFAULT_ITEM_COL])
        test = pd.DataFrame(index=index).reset_index()
        test[DEFAULT_RATING_COL] = self.eval_engine.predict(test, model)
        test.sort_values(by=[DEFAULT_USER_COL, DEFAULT_RATING_COL], ascending=[True, False])
        test[DEFAULT_USER_COL] = test[DEFAULT_USER_COL].apply(lambda x: self.data.id2user[x])
        test[DEFAULT_ITEM_COL] = test[DEFAULT_ITEM_COL].apply(lambda x: self.data.id2item[x])

        return test