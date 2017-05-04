#!/usr/bin/env python
# Deep Learning for Computer Vision practical course WS 2016/17
# Rajat Jain
# Protein function prediction from 2D representation of 3D structure

import argparse
import csv
import glob
import logging
import math
import os
import gzip
import pandas as pd
import numpy as np

# Package Bio can be obtained from http://www.biopython.org
from Bio import PDB
from Bio.PDB import *


logging.basicConfig(level=logging.DEBUG)


class GetTorsion(object):
    """
    Calculate side-chain torsion angles (also known as dihedral or chi angles).
    Depends: Biopython (http://www.biopython.org)
    """


    chi_atoms = dict(
        chi1=dict(
            ARG=['N', 'CA', 'CB', 'CG'],
            ASN=['N', 'CA', 'CB', 'CG'],
            ASP=['N', 'CA', 'CB', 'CG'],
            CYS=['N', 'CA', 'CB', 'SG'],
            GLN=['N', 'CA', 'CB', 'CG'],
            GLU=['N', 'CA', 'CB', 'CG'],
            HIS=['N', 'CA', 'CB', 'CG'],
            ILE=['N', 'CA', 'CB', 'CG1'],
            LEU=['N', 'CA', 'CB', 'CG'],
            LYS=['N', 'CA', 'CB', 'CG'],
            MET=['N', 'CA', 'CB', 'CG'],
            PHE=['N', 'CA', 'CB', 'CG'],
            PRO=['N', 'CA', 'CB', 'CG'],
            SER=['N', 'CA', 'CB', 'OG'],
            THR=['N', 'CA', 'CB', 'OG1'],
            TRP=['N', 'CA', 'CB', 'CG'],
            TYR=['N', 'CA', 'CB', 'CG'],
            VAL=['N', 'CA', 'CB', 'CG1'],
        ),
        #altchi1=dict(
        #    VAL=['N', 'CA', 'CB', 'CG2'],
        #),
        chi2=dict(
            ARG=['CA', 'CB', 'CG', 'CD'],
            ASN=['CA', 'CB', 'CG', 'OD1'],
            ASP=['CA', 'CB', 'CG', 'OD1'],
            GLN=['CA', 'CB', 'CG', 'CD'],
            GLU=['CA', 'CB', 'CG', 'CD'],
            HIS=['CA', 'CB', 'CG', 'ND1'],
            ILE=['CA', 'CB', 'CG1', 'CD1'],
            LEU=['CA', 'CB', 'CG', 'CD1'],
            LYS=['CA', 'CB', 'CG', 'CD'],
            MET=['CA', 'CB', 'CG', 'SD'],
            PHE=['CA', 'CB', 'CG', 'CD1'],
            PRO=['CA', 'CB', 'CG', 'CD'],
            TRP=['CA', 'CB', 'CG', 'CD1'],
            TYR=['CA', 'CB', 'CG', 'CD1'],
        ),
        #altchi2=dict(
        #    ASP=['CA', 'CB', 'CG', 'OD2'],
        #    LEU=['CA', 'CB', 'CG', 'CD2'],
        #    PHE=['CA', 'CB', 'CG', 'CD2'],
        #    TYR=['CA', 'CB', 'CG', 'CD2'],
        #),
        chi3=dict(
            ARG=['CB', 'CG', 'CD', 'NE'],
            GLN=['CB', 'CG', 'CD', 'OE1'],
            GLU=['CB', 'CG', 'CD', 'OE1'],
            LYS=['CB', 'CG', 'CD', 'CE'],
            MET=['CB', 'CG', 'SD', 'CE'],
        ),
        chi4=dict(
            ARG=['CG', 'CD', 'NE', 'CZ'],
            LYS=['CG', 'CD', 'CE', 'NZ'],
        ),
        chi5=dict(
            ARG=['CD', 'NE', 'CZ', 'NH1'],
        ),
        phi=dict(
            BASE=[],
        ),
        psi=dict(
            BASE=[],
        )
    )

    angle_mat_depth_list = [
        #'altchi1_VAL',
        #'altchi2_ASP','altchi2_LEU','altchi2_PHE','altchi2_TYR',
        'chi1_ARG','chi1_ASN','chi1_ASP','chi1_CYS','chi1_GLN','chi1_GLU','chi1_HIS','chi1_ILE','chi1_LEU','chi1_LYS','chi1_MET','chi1_PHE','chi1_PRO','chi1_SER','chi1_THR','chi1_TRP','chi1_TYR','chi1_VAL',
        'chi2_ARG','chi2_ASN','chi2_ASP','chi2_GLN','chi2_GLU','chi2_HIS','chi2_ILE','chi2_LEU','chi2_LYS','chi2_MET','chi2_PHE','chi2_PRO','chi2_TRP','chi2_TYR',
        'chi3_ARG','chi3_GLN','chi3_GLU','chi3_LYS','chi3_MET',
        'chi4_ARG','chi4_LYS',
        'chi5_ARG',
        'phi_BASE','psi_BASE'
    ]

    angle_list = ['sin', 'cos']

    angle_mat_depth_keys = ['{}_{}'.format(pre, suff) for suff in angle_mat_depth_list for pre in angle_list]
    logging.info(len(angle_mat_depth_keys))

    one_hot_encoding_list = chi_atoms['chi1'].keys()
    one_hot_encoding_list.append('ALA')
    one_hot_encoding_list.append('GLY')
    angle_mat_depth_keys.extend(one_hot_encoding_list)

    default_chi = [1,2,3,4,5]


    # Function to get the shape of the distance matrix so we can call the memmap lazy
    def get_size(self, pdb_id, handle):
        parser = PDBParser()
        try:
            structure = parser.get_structure(pdb_id, handle)
        except IOError:
            return False
        ca_atom_length = 0
        for model in structure:
            for chain in model:
                for residue in chain:
                    # Skip heteroatoms
                    if residue.id[0] != " ":
                        continue
                    for atom in residue:
                        if 'CA' in atom.get_id():
                            ca_atom_length += 1

        return ca_atom_length

    def __init__(self, ec_class):
        """Set parameters and calculate torsion values"""
        # Configure chi
        self.ec_class = ec_class[0]
        chi = self.default_chi
        chi_names = list()
        for x in chi:
            reg_chi = "chi%s" % x
            if reg_chi in self.chi_atoms.keys():
                chi_names.append(reg_chi)
                #alt_chi = "altchi%s" % x
                #if alt_chi in self.chi_atoms.keys():
                #    chi_names.append(alt_chi)
            else:
                logging.warning("Invalid chi %s", x)
        self.chi_names = chi_names
        self.dih_names = ["phi", "psi"]
        self.fieldnames = ["id", "model", "chain", "resn", "resi"] + self.chi_names + self.dih_names
        logging.debug("Calculating chi angles: %s", ", ".join(chi_names))

        # Configure units (degrees or radians)
        units = "degrees"
        self.degrees = bool(units[0].lower() == "d")
        self.degrees = False
        if self.degrees:
            message = "Using degrees"
        else:
            message = "Using radians"
        logging.debug(message)

        # Load parser
        self.parser = PDB.PDBParser(QUIET=True)

        # Construct list of files

        self.basePath = '/usr/data/cvpr_shared/proteins/enzymes/EC2PDB/pdb'
        self.targetPath = '/usr/data/cvpr_shared/proteins/enzymes/EC2PDB/rotamers_mmap/'
        #folders = os.listdir(basePath)
        logging.info('Starting conversion :\nBasePath : {} \nTargetPath : {}'.format(self.basePath, self.targetPath))
        folders = [str(self.ec_class)]
        pdb_count = 0
        nchannels = 104
        dtype = np.float32
        for ec_pos1 in folders:  # [3].4.1.1
            logging.info('EC pos1 : ' + ec_pos1 + '\n')
            pos_1_path = self.basePath + '/' + ec_pos1  # basePath/3
            ec_pos2_list = os.listdir(pos_1_path)  # 3.[4].1.1
            for ec_pos2 in ec_pos2_list:
                logging.info('EC pos2 : ' + ec_pos2 + '\n')
                pos_2_path = pos_1_path + '/' + ec_pos2  # basePath/3/4
                ec_pos3_list = os.listdir(pos_2_path)  # 3.4.[1].1
                for ec_pos3 in ec_pos3_list:
                    logging.info('EC pos3 : ' + ec_pos3 + '\n')
                    pos_3_path = pos_2_path + '/' + ec_pos3  # basePath/3/4/1
                    ec_pos4_list = os.listdir(pos_3_path)  # 3.4.1.[1]
                    for ec_pos4 in ec_pos4_list:
                        logging.info('EC pos4 : ' + ec_pos4 + '\n')
                        pos_4_path = pos_3_path + '/' + ec_pos4  # basePath/3/4/1/1
                        proteins = os.listdir(pos_4_path)  # [3.4.1.1]/*.pdb
                        for protein in proteins:
                            proteinpath = pos_4_path + '/' + protein
                            suffix = ec_pos1 + '/' + ec_pos2 + '/' + ec_pos3 + '/' + ec_pos4 + '/'
                            targetpath2 = self.targetPath + suffix + protein[:4] + '.mem'
                            logging.info(targetpath2)
                            handle = gzip.open(proteinpath)
                            pdb_id = proteinpath[-11:-7]
                            protsize = self.get_size(pdb_id, handle)
                            memmap_tmp = []

                            if protsize is not False:
                                pdb_count += 1
                                if pdb_count % 100 == 0:
                                    logging.info(pdb_count)

                            if protsize is None or "1bav" in proteinpath:
                                continue
                                self.append_pdb_data(pdb_id, proteinpath, message="Error in fetching size.")
                                #Create the memmaps and a list of all the memmaps that are being written
                            else:
                                self.to_memmap_lazy(read_fn=lambda memmap_tmp: self.get_rotamers_data(pdb_id, proteinpath, memmap_tmp, protsize),
                                          memmap_filename=targetpath2, dtype=dtype, shape=(protsize, nchannels))

        logging.info('Conversion completed.')





    # Function to write and read the memmaps
    def to_memmap_lazy(self, read_fn, memmap_filename, **kwargs):
        """
        Read any data source using read_fn() into a memmap'ed ndarray.
        However, if the memmap file already exists, use the existing memmap file and ignore the source.
        read_fn() takes the target ndarray as an *argument* (not output!).
        Additional arguments to read_fn() can be passed as follows:
            `tomemmap_lazy(read_fn=lambda memmap_tmp: my_read_fn(arg1, arg2, memmap_tmp), ...)`
        Arguments to numpy.memmap can be passed as follows:
            `tomemmap_lazy(..., dtype=..., shape=..., ...)`
        """
        if not os.path.isfile(memmap_filename):
            # read into *.memmap.tmp file; rename into *.memmap only when finished

            # create directory if required
            dir_memmap = os.path.split(memmap_filename)[0]
            if not os.path.exists(dir_memmap):
                os.makedirs(dir_memmap)
            logging.info("File name is : " + memmap_filename)
            try:
                memmap_tmp = np.memmap(memmap_filename + '.tmp', mode='w+', **kwargs)
            except IOError:
                logging.error('Incorrect file name : {}'.format(memmap_filename))
                os.remove(memmap_filename + '.tmp')
                return
            out = read_fn(memmap_tmp)
            del memmap_tmp  # thus memmap_tmp.flush() not necessary
            if out is False:
                os.remove(memmap_filename + '.tmp')
                return
            os.rename(memmap_filename + '.tmp', memmap_filename)

        return np.memmap(memmap_filename, mode='r', **kwargs)

    # Function that calculates the distance between the atoms within a protein structure
    def get_rotamers_data(self, pdb_id, proteinpath, angles, protsize):

        fn = gzip.open(proteinpath)
        torsion_list = self.calculate_torsion(id, proteinpath)
        if not torsion_list:
            return False
        # Convert torsion to matrix
        if len(torsion_list) != protsize:
            msg = "Size mismatch. Torsion = {} , Carbon atoms = {}".format(len(torsion_list), protsize)
            logging.info(msg)
            self.append_pdb_data(pdb_id, proteinpath, msg)
            return False
        else:
            angle_mat = self.get_anglemat(id, torsion_list)
            logging.info(angle_mat.shape)
            # write on distances
            angles[:] = angle_mat[:].copy()
            return True

    def append_pdb_data(self, pdb_id, proteinpath, message):
        file_path = '{}incorrect_torsion_{}.txt'.format(self.basePath[:-3], self.ec_class)
        with open(file_path, "a") as pdb_fault_file:
            line_to_add = "{} : {} . {} \n".format(pdb_id, proteinpath, message)
            pdb_fault_file.write(line_to_add)

    def get_anglemat(self, id, torsion_list):
        '''
        Calculate angle matrix from the torsion list
        Size should be L*104 to be copied L more times so as to synchronize with distance matrices shape.
        Input to the network will be of shape (L*L*208 after replication)
        :param id:
        :param torsion_list:
        :return: A matrix of size len(torsion_list) * len(angle_mat_depth_keys)  which contains values of chi angles filled accordingly
        '''
        angle_mat = np.zeros((len(torsion_list), len(self.angle_mat_depth_keys)))
        #logging.info(angle_mat.shape

        for index, res_data in enumerate(torsion_list):
            resname = res_data['resn']
            # One hot encoding
            if resname not in self.angle_mat_depth_keys:
                logging.info("Skipping resname = {} for id = {}".format(resname, id))
                continue
            else:
                angle_mat[index][self.angle_mat_depth_keys.index(resname)] = 1
            if resname in ("ALA", "GLY"): # Since all values are zero anyway, stored these AA just to match the size with distance matrix
                resname = 'BASE'
                self.fill_data('phi',angle_mat,index,res_data, resname)
                self.fill_data('psi', angle_mat, index, res_data, resname)
                continue
            for i,chi_n in enumerate(self.chi_atoms.keys()):
                self.fill_data(chi_n, angle_mat, index, res_data, resname)
        return angle_mat

    def fill_data(self, chi_n, angle_mat, index, res_data, resname):
        val = res_data[chi_n]
        if '' != str(val):
            if chi_n in ['phi', 'psi']:
                resname = 'BASE'
            sin_depth_index_name = 'sin_' + chi_n + '_' + resname
            cos_depth_index_name = 'cos_' + chi_n + '_' + resname
            # Now fill the angle matrix
            angle_mat[index][self.angle_mat_depth_keys.index(sin_depth_index_name)] = np.sin(val)
            angle_mat[index][self.angle_mat_depth_keys.index(cos_depth_index_name)] = np.cos(val)
            # logging.info(angle_mat[index,:]

    def calculate_torsion(self, id, fn):
        """Calculate side-chain torsion angles for given file"""

        torsion_list = list()
        structure = self.parser.get_structure(id, gzip.open(fn))
        for model in structure:
            model_name = model.id
            for chain in model:
                chain_name = chain.id
                polypeptides = PDB.PPBuilder().build_peptides(chain)
                dihedral_dict = self.get_phi_psi(polypeptides)
                for res in chain:
                    # Skip heteroatoms
                    if res.id[0] != " ":
                        continue
                    res_name = res.resname
                    phi = ''
                    psi = ''
                    dih_list = [phi, psi]
                    resi = "{0}{1}".format(res.resname, res.id[1])
                    # Fill in empty angles for these two AA so as to match size with distance matrix
                    if res_name in ("ALA", "GLY"):
                        if res_name == "GLY" and ('N' not in res or 'C' not in res):
                            msg = "Missing N or C atom."
                            logging.info(msg)
                            self.append_pdb_data(id, fn, msg)
                            return []
                        chi_list_non = ['','','','','']
                        if resi in dihedral_dict.keys():
                            dih_list = dihedral_dict[resi]
                        row = [id, model_name, chain_name, res_name, 0] + chi_list_non + dih_list
                        torsion_list.append(dict(zip(self.fieldnames, row)))
                        continue
                    chi_list = [""] * len(self.chi_names)
                    for x, chi in enumerate(self.chi_names):
                        chi_res = self.chi_atoms[chi]
                        try:
                            atom_list = chi_res[res_name]
                        except KeyError:
                            continue
                        try:
                            vec_atoms = [res[a] for a in atom_list]
                        except KeyError:
                            chi_list[x] = '' #Add dummy value instead of Nan such that sin and cosine both be zero
                            continue
                        vectors = [a.get_vector() for a in vec_atoms]
                        angle = PDB.calc_dihedral(*vectors)
                        if self.degrees:
                            angle = math.degrees(angle)
                        chi_list[x] = angle

                    # Calculate phi psi
                    if resi in dihedral_dict.keys():
                        dih_list = dihedral_dict[resi]
                    row = [id, model_name, chain_name, res_name, resi] + chi_list + dih_list
                    torsion_list.append(dict(zip(self.fieldnames, row)))

        return torsion_list

    def get_phi_psi(self,polypeptides):
        phi_psi_dict = {}
        for poly_index, poly in enumerate(polypeptides):
            phi_psi = poly.get_phi_psi_list()
            for res_index, residue in enumerate(poly):
                res_name = "%s%i" % (residue.resname, residue.id[1])
                phi_psi_list = list(phi_psi[res_index])
                if phi_psi_list[0] is None:
                    phi_psi_list[0] = ''
                if phi_psi_list[1] is None:
                    phi_psi_list[1] = ''
                phi_psi_dict[res_name] = phi_psi_list
        return phi_psi_dict

    @classmethod
    def commandline(cls):
        desc = """Calculate main chain and side-chain torsion angles for PDB files.
                Angles with missing atoms will be nan."""
        a = argparse.ArgumentParser(description=desc)
        a.add_argument('--ec_class', metavar='N', type=int, choices=range(7), default=[3], nargs='+',
                       help='EC class to process')
        args = a.parse_args()
        c = cls(**vars(args))
        return c

if __name__ == "__main__":

    GetTorsion.commandline()
