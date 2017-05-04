# Deep Learning for Computer Vision practical course WS 2016/17
# Avinash Kumar / Rajat Jain
# Protein function prediction from 2D representation of 3D structure

import os.path
import numpy as np
DTYPE = np.float32


class Protein(object):
    """A class using Bio.PDB library to perform different operations on protein data.
    Attributes :
        pdbid: 4 character protein id
        distmat_path : location of mem mapped file of distance matrix
        rotamers_path : location of mem mapped file of torsion angles
    """

    def __init__(self, pdbid, protein_sizes_and_ECnumbers, id_path_map, target_classes):
        """
        Initialize protein.
        :param pdbid:
        :param protein_sizes_and_ECnumbers: combined pandas dataframe containg information about sizes of all proteins
        :param id_path_map: dictionary of pdbid vs distance matrix mem mapped file path
        :param target_classes: List of target classes - 3rd level classification
        """
        self.pdbid = pdbid
        self.rotamers_base_path = '/usr/data/cvpr_shared/proteins/enzymes/EC2PDB/rotamers_mmap/'
        self.distmat_base_path = '/usr/data/cvpr_shared/proteins/enzymes/EC2PDB/distmat_mmap/'
        if pdbid in id_path_map:
            self.distmat_path = id_path_map[pdbid] # '/usr/data/cvpr_shared/proteins/enzymes/EC2PDB/distmat_mmap/3/4/21/1/1qwd.mem'
        else:
            msg = "Protein.py# Distance memory mapped file does not exist. Skipping {}".format(self.pdbid)
            raise Exception(msg)
        self.rotamers_path = self.get_rotamers_path()
        self.label, self.label_idx = self.get_label(target_classes)
        self.distance_mat_shape = self.get_shape_distance_mat(protein_sizes_and_ECnumbers)
        self.distance_mat_depth = 4
        self.rotamers_mat_depth = 104

    def get_rotamers_path(self):
        """
        Calculate path of rotamers mem mapped files given the distance matrix mem file path
        distmat_path = <distmat_base_path> + <protein_ec_path> + <pdbid>.mem
        So distmat_base_path is of length 59
        Hence protein_ec_path is between 59:-8

        :return: rotamers_path
        """
        ec_path = self.distmat_path[59:-8]
        rotamers_path = self.rotamers_base_path + ec_path + self.pdbid + '.mem'
        if os.path.isfile(rotamers_path):
            return rotamers_path
        else:
            msg = "Protein.py# Rotamers memory mapped file does not exist. Skipping {}".format(self.pdbid)
            raise Exception(msg)

    def get_label(self, target_classes):
        """
        Parse the distance matrix path to get the enzyme classification label.
        :param target_classes
        :return: integer value of label
        """
        ec_path = self.distmat_path[59:-8] # "3/4/21/11/"
        levels = ec_path.split('/')
        class_label = '_'.join([levels[0], levels[1], levels[2]])
        if class_label in target_classes:
            return class_label, target_classes.index(class_label)
        else:
            msg = "Protein.py# Invalid label for distance matrix mem file : {}".format(self.distmat_path)
            raise Exception(msg)

    def get_shape_distance_mat(self, df):
        """
        Find the shape of distance matrix using dataframe of "/usr/prakt/w049/deep_learning/input_data/dict_<>.xlsx"
        The excel sheet has 5 columns 'pdbid', 'rows', 'cols', 'depth', 'label'
        :param df:
        :return: rows, cols
        """
        data = None
        if self.pdbid in df.values:
            data = df.loc[df['pdbid'] == self.pdbid]
        if data is not None:
            rows = data['rows'].values[0]
            cols = rows
            return rows, cols
        else:
            return None

    def get_rotamers_mmap(self):
        return np.memmap(self.rotamers_path, dtype=DTYPE,
                         mode='r', shape=(1, 1, self.distance_mat_shape[0],
                                         self.rotamers_mat_depth))

    def get_distmat_mmap(self):
        return np.memmap(self.distmat_path, dtype=DTYPE,
                         mode='r',shape=(1, self.distance_mat_shape[0],
                                         self.distance_mat_shape[1], 4))

    def get_input_map(self):
        """
        This function -
            1. reads distance matrix mmap and rotamers mmap
            2. reshapes and tiles the rotamers mmap both horizontally and vertically from 1*1*L*104 to 1*L*L*208
            3. Concatenate both memmaps and transpose the axes to feed to the network
        :return: input_map to be given to the network
        """
        d_mmap = self.get_distmat_mmap()
        r_mmap = self.get_rotamers_mmap()
        r_mmap_orient_1 = r_mmap.reshape(1, self.distance_mat_shape[0], -1)
        r_mmap_orient_2 = r_mmap.reshape(self.distance_mat_shape[0], 1, -1)

        r_mmap_t_1 = np.tile(r_mmap_orient_1, (1, self.distance_mat_shape[0], 1, 1))
        r_mmap_t_2 = np.tile(r_mmap_orient_2, (1, 1, self.distance_mat_shape[0], 1))

        input_map = np.concatenate((d_mmap, r_mmap_t_1, r_mmap_t_2), axis = 3)
        input_map = np.transpose(input_map, (0, 3, 1, 2)) # Swap axes to feed to network
        return input_map
