from __future__ import print_function, division
import os
import os.path
import torch
import numpy as np
import pandas as pd
import math
import re
import pdb
import pickle
from sklearn.preprocessing import LabelEncoder
from scipy import stats

from torch.utils.data import Dataset
import h5py


"""
Adopted from MCAT: https://github.com/mahmoodlab/MCAT
"""
class Generic_WSI_Classification_Dataset(Dataset):
    def __init__(self,
        csv_path = './dataset_csv/tcga_gbmlgg_all_clean.csv.zip',
        mode = 'omic', 
        apply_sig = True,
        shuffle = False, 
        seed = 7, 
        print_info = True, 
        ignore=[],
        patient_strat=False, 
        label_col = None, 
        filter_dict = {}, 
        eps=1e-6):
        r"""
        Generic_WSI_Survival_Dataset 

        Args:
            csv_file (string): Path to the csv file with annotations.
            shuffle (boolean): Whether to shuffle
            seed (int): random seed for shuffling the data
            print_info (boolean): Whether to print a summary of the dataset
            label_dict (dict): Dictionary with key, value pairs for converting str labels to int
            ignore (list): List containing class labels to ignore
        """
        self.custom_test_ids = None
        self.seed = seed
        self.print_info = print_info
        self.patient_strat = patient_strat # ?
        self.train_ids, self.val_ids, self.test_ids  = (None, None, None)
        self.data_dir = None

        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(slide_data)


        slide_data = pd.read_csv(csv_path, compression='zip', header=0, index_col=0, sep=',',  low_memory=False)   # read genomic data
        if 'case_id' not in slide_data:
            slide_data.insert(0,'case_id', slide_data.index)
        # if 'case_id' not in slide_data:
        #     slide_data.index = slide_data.index.str[:12]    # org index -> case_id
        #     slide_data['case_id'] = slide_data.index
        slide_data = slide_data.reset_index(drop=True)  # reset index

        self.label_col = 'oncotree_code'
        label = np.array(slide_data['oncotree_code'])
        encoder = LabelEncoder()
        encoder.fit(label)
        # Generate the numerical labels.
        label = encoder.transform(label)
        slide_data['oncotree_code'] = label     # encoder contain names
        self.encoder = encoder

        patients_df = slide_data.drop_duplicates(['case_id']).copy()    # non-repeat data

        for patient in slide_data.index:           # modify slide_id
            full = slide_data.loc[patient, "slide_id"]
            slide_data.at[patient, "slide_id"] = full[:13]+full[20:23]
        # print(slide_data['slide_id'][0])
        for patient in patients_df.index:           # modify slide_id
            full = patients_df.loc[patient, "slide_id"]
            patients_df.at[patient, "slide_id"] = full[:13]+full[20:23]
        # print(patients_df['slide_id'][0])


        patient_dict = {}
        slide_data = slide_data.set_index('case_id')    #使用现有列设置 DataFrame 索引
        for patient in patients_df['case_id']:
            slide_ids = slide_data.loc[patient, 'slide_id']     # WSI id
            if isinstance(slide_ids, str):
                slide_ids = np.array(slide_ids).reshape(-1)
            else:
                slide_ids = slide_ids.values
            patient_dict.update({patient:slide_ids})    # patient id 对应(许多)WSI

        self.patient_dict = patient_dict    # patient id 对应 WSI
        print("self.patient_dict", self.patient_dict.keys)
    
        slide_data = patients_df
        slide_data.reset_index(drop=True, inplace=True)
        # slide_data = slide_data.assign(slide_id=slide_data['case_id']) 

        self.num_classes=len(np.unique(label))
        # patients_df = slide_data.drop_duplicates(['case_id'])
        self.patient_data = {'case_id':patients_df['case_id'].values, 'oncotree_code':patients_df[self.label_col].values}

        self.slide_data = slide_data
        print('self.slide_data.shape', self.slide_data.shape)
        self.metadata = slide_data.columns[:9]
        self.mode = mode
        self.cls_ids_prep()

        if print_info:
            self.summarize()

        ### Signatures
        self.apply_sig = apply_sig
        if self.apply_sig:
            self.signatures = pd.read_csv('./dataset_csv/signatures.csv')
        else:
            self.signatures = None

        if print_info:
            self.summarize()


    def cls_ids_prep(self):
        self.patient_cls_ids = [[] for i in range(self.num_classes)]   # patient ids of each class
        for i in range(self.num_classes):
            self.patient_cls_ids[i] = np.where(self.patient_data['oncotree_code'] == i)[0]

        self.slide_cls_ids = [[] for i in range(self.num_classes)]     # slide ids of each class
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['oncotree_code'] == i)[0]


    def patient_data_prep(self):
        patients = np.unique(np.array(self.slide_data['case_id'])) # get unique patients
        patient_labels = []
        
        for p in patients:
            locations = self.slide_data[self.slide_data['case_id'] == p].index.tolist()
            assert len(locations) > 0
            label = self.slide_data['oncotree_code'][locations[0]] # get patient label
            patient_labels.append(label)
        
        self.patient_data = {'case_id':patients, 'oncotree_code':np.array(patient_labels)}


    @staticmethod
    def df_prep(data, n_bins, ignore, label_col):
        mask = data[label_col].isin(ignore)
        data = data[~mask]
        data.reset_index(drop=True, inplace=True)
        disc_labels, bins = pd.cut(data[label_col], bins=n_bins)
        return data, bins

    def __len__(self):
        if self.patient_strat:
            return len(self.patient_data['case_id'])
        else:
            return len(self.slide_data)

    def summarize(self):
        print("label column: {}".format(self.label_col))
        print("number of classes: {}".format(self.num_classes))
        print("slide-level counts: ", '\n', self.slide_data['oncotree_code'].value_counts(sort = False))
        for i in range(self.num_classes):
            print('Slide-LVL; Number of samples registered in class %d: %d' % (i, self.slide_cls_ids[i].shape[0]))


    def get_split_from_df(self, all_splits: dict, split_key: str, scaler=None):
        split = all_splits[split_key]
        split = split.dropna().reset_index(drop=True)
        print("Split:", split.tolist())
        # print(self.slide_data['slide_id'][0])

        if len(split) > 0:
            mask = self.slide_data['case_id'].isin(split.tolist())
            df_slice = self.slide_data[mask].copy().reset_index(drop=True)
            split = Generic_Split(df_slice, 
                                metadata=self.metadata, 
                                mode=self.mode,
                                signatures=self.signatures, 
                                data_dir=self.data_dir, 
                                label_col=self.label_col, 
                                patient_dict=self.patient_dict,
                                num_classes=self.num_classes)
        else:
            split = None
        
        return split


    def return_splits(self, from_id: bool=True, csv_path: str=None):
        if from_id:
            raise NotImplementedError
        else:
            assert csv_path 
            all_splits = pd.read_csv(csv_path)
            train_split = self.get_split_from_df(all_splits=all_splits, split_key='train')
            val_split = self.get_split_from_df(all_splits=all_splits, split_key='val')

        return train_split, val_split#, test_split


    def get_list(self, ids):
        return self.slide_data['slide_id'][ids]

    def getlabel(self, ids):
        return self.slide_data['label'][ids]

    def __getitem__(self, idx):
        return None


class Generic_MIL_Dataset(Generic_WSI_Classification_Dataset):
    def __init__(self,
        data_dir, 
        **kwargs):
        super(Generic_MIL_Dataset, self).__init__(**kwargs)
        self.data_dir = data_dir
        self.use_h5 = False

    def load_from_h5(self, toggle):
        self.use_h5 = toggle

    def __getitem__(self, idx):
        case_id = self.slide_data['case_id'][idx]
        label = self.slide_data['oncotree_code'][idx]
        slide_ids = self.patient_dict[case_id]

        data_dir = self.data_dir
        
        if not self.use_h5:
            if self.data_dir:
                if self.mode == 'path':
                    path_features = []
                    for slide_id in slide_ids:
                        #TODO: file_name of WSI
                        wsi_path = os.path.join(data_dir, '{}.pt'.format(slide_id.rstrip('.svs')))
                        if not os.path.isfile(wsi_path):
                            # print("Not exist: ", slide_id)
                            continue 
                        wsi_bag = torch.load(wsi_path)
                        path_features.append(wsi_bag)
                        continue
                    if len(path_features) == 0:
                        return (torch.Tensor(),torch.Tensor(),-1)
                    path_features = torch.cat(path_features, dim=0)
                    return (path_features, torch.zeros((1,1)), label)

                elif self.mode == 'pathomic':   # with genomic_features
                    path_features = []
                    for slide_id in slide_ids:
                        wsi_path = os.path.join(data_dir, '{}.pt'.format(slide_id.rstrip('.svs')))
                        if not os.path.isfile(wsi_path):
                            # print("Not exist: ", slide_id)
                            continue 
                        wsi_bag = torch.load(wsi_path)
                        path_features.append(wsi_bag)
                        continue
                    if len(path_features) == 0:
                        return (torch.Tensor(),torch.Tensor(),-1)
                    path_features = torch.cat(path_features, dim=0)
                    genomic_features = torch.tensor(self.genomic_features.iloc[idx])
                    return (path_features, genomic_features, label)

                elif self.mode == 'coattn':
                    path_features = []
                    for slide_id in slide_ids:
                        wsi_path = os.path.join(data_dir, '{}.pt'.format(slide_id.rstrip('.svs')))
                        # print(wsi_path)
                        if not os.path.isfile(wsi_path):
                            # print("Not exist: ", slide_id)
                            continue 
                        wsi_bag = torch.load(wsi_path)
                        path_features.append(wsi_bag)
                        continue
                    if len(path_features) == 0:
                        # print("Empty: ", slide_ids)
                        return (torch.Tensor(),torch.Tensor(),torch.Tensor(),
                                torch.Tensor(),torch.Tensor(),torch.Tensor(),
                                torch.Tensor(),-1)
                    path_features = torch.cat(path_features, dim=0)
                    omic1 = torch.tensor(self.genomic_features[self.omic_names[0]].iloc[idx])
                    omic2 = torch.tensor(self.genomic_features[self.omic_names[1]].iloc[idx])
                    omic3 = torch.tensor(self.genomic_features[self.omic_names[2]].iloc[idx])
                    omic4 = torch.tensor(self.genomic_features[self.omic_names[3]].iloc[idx])
                    omic5 = torch.tensor(self.genomic_features[self.omic_names[4]].iloc[idx])
                    omic6 = torch.tensor(self.genomic_features[self.omic_names[5]].iloc[idx])
                    return (path_features, omic1, omic2, omic3, omic4, omic5, omic6, label)
                else:
                    raise NotImplementedError('Mode [%s] not implemented.' % self.mode)
                ### <--
            else:
                return slide_ids, label


class Generic_Split(Generic_MIL_Dataset):
    def __init__(self, slide_data, metadata, mode, signatures=None, data_dir=None,
                label_col=None, patient_dict=None, num_classes=6):
        self.use_h5 = False
        self.slide_data = slide_data
        self.metadata = metadata
        self.mode = mode
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.label_col = label_col
        self.patient_dict = patient_dict
        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['oncotree_code'] == i)[0]

        ### --> Initializing genomic features in Generic Split
        print("Org Shape", self.slide_data.shape)
        self.genomic_features = self.slide_data.drop(self.metadata, axis=1)
        self.signatures = signatures
        self.omic_input_size = len(self.genomic_features.columns)

        def series_intersection(s1, s2):
            return pd.Series(list(set(s1) & set(s2)))   # 交集

        if self.signatures is not None:
            self.omic_names = []                # a list of series, each contain corresponding genomic_features
            for col in self.signatures.columns:
                omic = self.signatures[col].dropna().unique()
                omic = np.concatenate([omic+mode for mode in ['_mut', '_cnv', '_rnaseq']])
                omic = sorted(series_intersection(omic, self.genomic_features.columns))
                self.omic_names.append(omic)
            self.omic_sizes = [len(omic) for omic in self.omic_names]
        print("Shape", self.genomic_features.shape)
        ### <--

    def __len__(self):
        return len(self.slide_data)
