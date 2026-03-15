from __future__ import division, print_function, absolute_import
import os
import os.path as osp
import glob
import re
import pandas as pd
import warnings
from ..dataset import ImageDataset
import cv2
import numpy as np
from PIL import Image
from torchreid.utils import read_image

class TVRID(ImageDataset):
    """TVRID Competition Dataset.
    
    Data layout:
    - root/train/passage_path/...
    - root/test_public/passage_path/...
    
    Args:
        root (str): path to 'data/DB_extracted'
        tvrid_track (str): 'rgb', 'depth', or 'cross'. Controls which images are loaded.
    """
    dataset_dir = '' # uses root directly

    def __init__(self, root='', tvrid_track='rgb', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = self.root
        self.tvrid_track = tvrid_track
        self.apply_jet = (tvrid_track in ['depth', 'cross'])
        
        # Paths
        self.train_csv = osp.join(self.root, 'train_labels.csv')
        self.test_csv = osp.join(self.root, 'public_test_labels.csv')
        self.train_subdir = osp.join(self.root, 'train')
        self.test_subdir = osp.join(self.root, 'test_public')
        
        required_files = [self.train_csv, self.test_csv, self.train_subdir, self.test_subdir]
        self.check_before_run(required_files)
        
        train = self.process_csv(self.train_csv, self.train_subdir, is_train=True)
        # For competition, 'query' and 'gallery' are defined in the test csv
        # The test csv has columns: gallery_id, path.
        # But wait, standard torchreid expects split query/gallery.
        # The competition submission format requires ranking gallery items for query items.
        # Actually, in this competition, the 'test' set is likely a gallery and we query it?
        # Re-reading README: "query_gallery_id" ... "rankings"
        # Usually checking 'eval_generate.py' reveals how query/gallery are structured.
        # But for *training* dataset initialization, we usually provide a validation split.
        # The README says: "public_test_labels.csv is intended for ranking generation... must not be used as ... validation".
        # So for 'query' and 'gallery' in ImageDataset (which are usually val/test), we should use a hold-out from train or just empty if we only do training.
        # However, to allow 'test' method to work, we can load public test.
        # But standard torchreid 'test' expects labeled query/gallery. 
        # The public test has no IDs (pid)? "subject-disjoint".
        # public_test_labels.csv has 'gallery_id'. Is that the PID?
        # README: "rankings_rgb.csv ... query_gallery_id... gallery_id...".
        # It seems 'gallery_id' is the unique ID of the *video sequence* (passage)? No.
        # Let's inspect 'eval_generate.py' later.
        # For now, I will load 'train' properly. 
        # Valid/Query/Gallery will be populated but might not have ground truth PIDs if using public test.
        # WE WILL SPLIT TRAIN for validation if needed, or just load all train.
        
        # To be safe and compatible with standard torchreid training:
        # I'll simply load the full training set into 'train'.
        # I will leave query/gallery empty or just put a dummy set if strictly required.
        # Torchreid requires query and gallery to not be empty usually? 
        # Actually, if we just run 'train_tvrid.py' with only training, it's fine.
        # But usually we want some validation.
        # I will implement a rudimentary split of train set for validation (query/gallery) 
        # just so the engine doesn't crash, but user can override.
        # Actually, I'll essentially make 'train' = all train data.
        # And 'query'/'gallery' = subset of train data (e.g. last few IDs) if we want validation.
        # But for now, let's just load everything into train.
        
        query = []
        gallery = []
        
        # We can implement a naive split here or let the user do it via CSVs. 
        # I'll just load all train data.
        
        super(TVRID, self).__init__(train, query, gallery, **kwargs)

    def process_csv(self, csv_path, subdir, is_train=True):
        df = pd.read_csv(csv_path)
        data = []
        
        # Required columns for train: person_id, cam_id, path
        # If test csv doesn't have person_id, we can't use it for training anyway.
        
        if 'person_id' not in df.columns:
            # likely test csv
            return []

        # Relabel PIDs to be 0-indexed contiguous
        if is_train:
            unique_pids = sorted(df['person_id'].unique())
            pid2label = {pid: i for i, pid in enumerate(unique_pids)}
        
        # Group by passage (path) to avoid re-scanning too much, but we need individual frames
        for idx, row in df.iterrows():
            pid = int(row['person_id'])
            if is_train:
                pid = pid2label[pid]
            
            camid = int(row['cam_id']) - 1 # 1-based to 0-based
            passage_path = row['path'].replace('\\', os.sep)
            full_passage_dir = osp.join(subdir, passage_path)
            
            # Find images
            images = self._find_images(full_passage_dir)
            
            for img_path in images:
                data.append((img_path, pid, camid))
                
        return data

    def __getitem__(self, index):
        img_path, pid, camid, dsetid = self.data[index]
        
        # Check if it is a depth image
        is_depth = ('_depth.png' in img_path) or ('_D.png' in img_path)
        
        if self.apply_jet and is_depth:
            # Custom loading for depth: Apply JET colormap
            if not osp.exists(img_path):
                raise IOError('"{}" does not exist'.format(img_path))
                
            # Load as grayscale (CV2 loads as BGR by default, use IMREAD_GRAYSCALE)
            img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img_gray is None:
                 raise IOError('"{}" could not be read'.format(img_path))
                 
            # Apply JET colormap (produces BGR)
            img_color = cv2.applyColorMap(img_gray, cv2.COLORMAP_JET)
            
            # Convert BGR to RGB (PIL/Torchreid expects RGB)
            img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            img = Image.fromarray(img_color)
            
        else:
            # Standard loading
            img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img) # Use self.transform directly as in base ImageDataset
            
        item = {
            'img': img,
            'pid': pid,
            'camid': camid,
            'impath': img_path,
            'dsetid': dsetid
        }
        return item

    def _find_images(self, passage_dir):
        # Logic to find images based on track
        # track = 'rgb' -> *_RGB.png
        # track = 'depth' -> *_depth.png (or similar)
        # track = 'cross' -> both?
        
        # Based on utils/data.py:
        # rgb_stems = {f.split("_RGB")[0] for f in files if "_RGB" in f}
        # depth_stems = {f.split("_depth")[0] for f in files if "_depth" in f}
        
        if not osp.exists(passage_dir):
            return []
            
        files = os.listdir(passage_dir)
        valid_files = []
        
        stems = set()
        # simplified scanning
        for f in files:
            if self.tvrid_track in ['rgb', 'cross'] and '_RGB.png' in f:
                valid_files.append(osp.join(passage_dir, f))
            if self.tvrid_track in ['depth', 'cross'] and ('_depth.png' in f or '_D.png' in f):
                # Note: utils/data.py handles depth loading specially (1 channel). 
                # Torchreid standard loader expects 3 channel images.
                # The user's provided starter kit converts depth to tensor.
                # Here we list paths. The standard loader will load with PIL.convert('RGB').
                # Valid depth images (1 channel) loaded as RGB look greyish/black.
                # This is a standard simple trick.
                valid_files.append(osp.join(passage_dir, f))
                
        return sorted(valid_files)
