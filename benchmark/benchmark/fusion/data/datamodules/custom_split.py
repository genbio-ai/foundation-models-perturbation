import lightning as L
from ..datasets import (
    TahoeDEGDataset,
    EssentialPseudobulkDataset,
    EssentialDEGDataset,
    SciplexDEGDataset,
)
from sklearn.model_selection import train_test_split
from typing import List
import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import anndata as ad

from benchmark import BenchmarkTask

class LeaderboardSplitDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_type: str,
        split_key: str,
        fold: int,
        condition_encoders: List,
        batch_size: int,
        num_workers: int,
        validation_split_ratio: float,
        seed: int,
        **dataset_kwargs,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_key = split_key
        self.fold = fold

        self.validation_split_ratio = validation_split_ratio
        self.seed = seed

        # Load dataset
        if dataset_type == "tahoe_deg":

            train_list, test_list = [], []

            for cell_line in tqdm([
                'CVCL_0023',
                'CVCL_0028',
                'CVCL_0069',
                'CVCL_0099',
                'CVCL_0131',
                'CVCL_0152',
                'CVCL_0179',
                'CVCL_0218',
                'CVCL_0292',
                'CVCL_0293',
                'CVCL_0320',
                'CVCL_0332',
                'CVCL_0334',
                'CVCL_0359',
                'CVCL_0366',
                'CVCL_0371',
                'CVCL_0397',
                'CVCL_0399',
                'CVCL_0428',
                'CVCL_0459',
                'CVCL_0480',
                'CVCL_0504',
                'CVCL_0546',
                'CVCL_1055',
                'CVCL_1056',
                'CVCL_1094',
                'CVCL_1097',
                'CVCL_1098',
                'CVCL_1119',
                'CVCL_1125',
                'CVCL_1239',
                'CVCL_1285',
                'CVCL_1381',
                'CVCL_1478',
                'CVCL_1495',
                'CVCL_1517',
                'CVCL_1547',
                'CVCL_1550',
                'CVCL_1635',
                'CVCL_1666',
                'CVCL_1693',
                'CVCL_1717',
                'CVCL_1724',
                'CVCL_1731',
                'CVCL_C466',
            ]):
                task = BenchmarkTask(
                    name="expression/pert-prediction-tahoe-deg",
                    fold=f"{cell_line}.{fold}",
                )

                train, test = task.setup()

                train.obs['smiles'] = train.obs['canonical_smiles']
                test.obs['smiles'] = test.obs['canonical_smiles']
                train.obs['cell_line'] = cell_line
                test.obs['cell_line'] = cell_line
                train.obs['pubchem_cid'] = train.obs['pert_id'].astype(np.int64)
                test.obs['pubchem_cid'] = test.obs['pert_id'].astype(np.int64)

                train_list.append(train.copy())
                test_list.append(test.copy())

            
            self.train = ad.concat(train_list)
            self.test = ad.concat(test_list)
            

            self.train_dataset = TahoeDEGDataset(
                condition_encoders=condition_encoders, adata=self.train, **dataset_kwargs
            )
            self.test_dataset = TahoeDEGDataset(
                condition_encoders=condition_encoders, adata=self.test, **dataset_kwargs
            )
        elif dataset_type == "essential_deg":

            train_list, test_list = [], []

            for cell_line in tqdm(["Hep-G2", "Jurkat", "K-562", "hTERT-RPE1"]):
                task = BenchmarkTask(
                    name="expression/pert-prediction-essential-deg",
                    fold=f"{cell_line}.{fold}",
                )

                train, test = task.setup()

                train.obs['cell_line'] = cell_line
                test.obs['cell_line'] = cell_line

                train.obs['gene_id'] = train.obs['pert_id']
                test.obs['gene_id'] = test.obs['pert_id']

                train_list.append(train.copy())
                test_list.append(test.copy())

            
            self.train = ad.concat(train_list)
            self.test = ad.concat(test_list)            

            self.train_dataset = EssentialDEGDataset(
                condition_encoders=condition_encoders, adata=self.train, **dataset_kwargs
            )
            self.test_dataset = EssentialDEGDataset(
                condition_encoders=condition_encoders, adata=self.test, **dataset_kwargs
            )
        elif dataset_type == "essential_lfc":

            train_list, test_list = [], []

            for cell_line in tqdm(["Hep-G2", "Jurkat", "K-562", "hTERT-RPE1"]):
                task = BenchmarkTask(
                    name="expression/pert-prediction-essential-regression-all-perturbations",
                    fold=f"{cell_line}.{fold}",
                )

                train, test = task.setup()

                train.obs['cell_line'] = cell_line
                test.obs['cell_line'] = cell_line

                train.obs['gene_id'] = train.obs['pert_id']
                test.obs['gene_id'] = test.obs['pert_id']

                train_list.append(train.copy())
                test_list.append(test.copy())

            
            self.train = ad.concat(train_list)
            self.test = ad.concat(test_list)


            self.train_dataset = EssentialPseudobulkDataset(
                condition_encoders=condition_encoders, adata=self.train, **dataset_kwargs
            )
            self.test_dataset = EssentialPseudobulkDataset(
                condition_encoders=condition_encoders, adata=self.test, **dataset_kwargs
            )
        elif dataset_type == "sciplex_deg":

            train_list, test_list = [], []

            for cell_line in tqdm(['A549', 'K562', 'MCF7']):
                task = BenchmarkTask(
                    name="expression/pert-prediction-sciplex3-deg",
                    fold=f"{cell_line}.{fold}",
                )

                train, test = task.setup()

                train_list.append(train.copy())
                test_list.append(test.copy())

            
            self.train = ad.concat(train_list)
            self.test = ad.concat(test_list)
            

            self.train_dataset = SciplexDEGDataset(
                condition_encoders=condition_encoders, adata=self.train, **dataset_kwargs
            )
            self.test_dataset = SciplexDEGDataset(
                condition_encoders=condition_encoders, adata=self.test, **dataset_kwargs
            )
        else:
            raise ValueError("Dataset unknown")

    def setup(self, stage: str):

        train_values = self.train.obs[self.split_key].unique()
        if stage == "fit":
            # Split into train and val
            train_values, val_values = train_test_split(
                train_values,
                test_size=self.validation_split_ratio,
                random_state=self.seed,
                shuffle=True,
            )

            # Get the train and val indices
            split_key_vals = self.train_dataset.adata.obs[self.split_key]
            self.train_idx = np.argwhere(split_key_vals.isin(train_values)).ravel()
            self.val_idx = np.argwhere(split_key_vals.isin(val_values)).ravel()

        if stage == "predict":
            # Get the test indices
            split_key_vals = self.test.obs[self.split_key]

    def train_dataloader(self):
        return DataLoader(
            Subset(self.train_dataset, self.train_idx),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            Subset(self.train_dataset, self.val_idx),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True,
        )