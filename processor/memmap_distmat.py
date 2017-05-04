# Deep Learning for Computer Vision practical course
# Rajat Jain and Avinash Singh
# Magda Paschali and Merkouris Simos
# Function from known structure

import os
import math
import numpy as np
import gzip
import matplotlib.pyplot as plt
from Bio.PDB import *
import csv
import argparse


# Function to write and read the memmaps
def tomemmap_lazy(read_fn, memmap_filename, **kwargs):
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
        print "File name is : " + memmap_filename
        try:
            memmap_tmp = np.memmap(memmap_filename + '.tmp', mode='w+', **kwargs)
        except IOError:
            print "Incorrect protein : {}".format(memmap_filename)
            os.remove(memmap_filename + '.tmp')
            return
        out = read_fn(memmap_tmp)
        del memmap_tmp  # thus memmap_tmp.flush() not necessary
        if out is False:
            os.remove(memmap_filename + '.tmp')
            return
        os.rename(memmap_filename + '.tmp', memmap_filename)

    return np.memmap(memmap_filename, mode='r', **kwargs)


def append_pdb_data(pdb_id, proteinpath, message):
    file_path = '{}faulty_ids_distmat_{}.txt'.format(basePath[:-3], ec_class)
    with open(file_path, "a") as pdb_fault_file:
        pdb_fault_file.write(pdb_id + " : " + proteinpath + " . " + message + "\n")


# Function that calculates the distance between the atoms within a protein structure
def get_dist_mat(pdb_id, proteinpath, distances):

    structure = parser.get_structure(pdb_id, gzip.open(proteinpath))
    pi = math.pi

    # find Ca and Cb atoms
    all_CA = []
    all_CB = []
    for model in structure:
        for chain in model:
            for residue in chain:
                # Skip heteroatoms
                if residue.id[0] != " ":
                    continue
                # if it is a GLY residue we need to create a 'Virtual' Cb atom
                if residue.get_resname() == "GLY":
                    if 'N' in residue:
                        n = residue['N'].get_vector()
                    else:
                        append_pdb_data(pdb_id, proteinpath, message="N not in residue")
                        return False
                    if 'C' in residue:
                        c = residue['C'].get_vector()
                    else:
                        append_pdb_data(pdb_id, proteinpath, message="C not in residue")
                        return False

                    ca = residue['CA'].get_vector()
                    # center at origin
                    n = n - ca
                    c = c - ca
                    # find rotation matrix that rotates n
                    # -120 degrees along the ca-c vector
                    rot = rotaxis(-pi * 120.0 / 180.0, c)
                    # apply rotation to ca-n vector
                    cb_at_origin = n.left_multiply(rot)
                    # put on top of ca atom
                    cb = cb_at_origin + ca
                    # create Cb from a copy
                    fakeCB = residue['N'].copy()
                    # vector to array
                    cb = cb.get_array()
                    # add the coords we calculated
                    fakeCB.set_coord(cb)
                    all_CB.append(fakeCB)
                for atom in residue:
                    # Also works with -> if atom.get_id()=="CA":
                    if 'CA' in atom.get_id():

                        all_CA.append(atom)
                        # elif atom.get_id()=="CB":
                    elif 'CB' in atom.get_id() and residue.get_resname() != "GLY":
                        all_CB.append(atom)
    all_CB_len = len(all_CB)
    all_CA_len = len(all_CA)
    if all_CA_len != all_CB_len:
        msg = "Incorrect calculation : CA=" + str(all_CA_len) + " , CB=" + str(
            all_CB_len) + " . Skipping pdb."
        print msg
        append_pdb_data(pdb_id, proteinpath, message=msg)
        return False
    CaCb = all_CA + all_CB
    atLen = len(CaCb)
    if (atLen > 12000):
        print "M size too large : " + str(atLen) + " , Skipping : " + proteinpath + "\n"
        append_pdb_data(pdb_id, proteinpath, message="Size too large : " + str(atLen))
        return False
        # calculate distances
    distMat = np.zeros((atLen, atLen))
    for i in range(atLen):
        for j in range(i + 1, atLen):
            dist = CaCb[i] - CaCb[j]
            distMat[i, j] = dist
            distMat[j, i] = distMat[i, j]
    # Change the shape from 2Lx2L to 1x4xLxL
    caca = distMat[0:len(all_CA), 0:len(all_CA)]
    cacb = distMat[0:len(all_CA), 0 + len(all_CA):2 * len(all_CA)]
    cbca = distMat[0 + len(all_CA):2 * len(all_CB), 0:len(all_CB)]
    cbcb = distMat[0 + len(all_CA):2 * len(all_CB), 0 + len(all_CA):2 * len(all_CA)]

    # write on distances
    distances2 = np.concatenate(
        (caca[..., np.newaxis], cacb[..., np.newaxis], cbca[..., np.newaxis], cbcb[..., np.newaxis]), axis=2)
    distances[:] = distances2[:].copy()
    return True


# Function to get the shape of the distance matrix so we can call the memmap lazy
def get_size(pdb_id, handle):
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


def add_to_csv(pdb_id_shape_class):
    csv_path = '/usr/prakt/w049/dict_{}.csv'.format(ec_class)
    with open(csv_path, 'wb') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in pdb_id_shape_class.items():
            writer.writerow([key, value])


# Execution starts here

# Get ec_class for which we want to compute the distance matrices/protein sizes
desc = """Calculate side-chain torsion angles for PDB files.
               Angles with missing atoms will be nan."""
parse = argparse.ArgumentParser(description=desc)
parse.add_argument('--ec_class', metavar='N', type=int, choices=range(7), default=[3], nargs='+',
                   help='EC class to process')
parse.add_argument('--size_only', dest='size_only', action='store_true',
                   help='If used , class will not work on distance matrices.\nRather just provide a .csv file with the sizes of all proteins')
args = parse.parse_args()
ec_class = args.ec_class[0]
size_only = args.size_only


dtype = np.float32
nchannels = 4

parser = PDBParser()
basePath = '/usr/data/cvpr_shared/proteins/enzymes/EC2PDB/pdb'
targetPath = '/usr/data/cvpr_shared/proteins/enzymes/EC2PDB/distmat_mmap/'
folders = os.listdir(basePath)
print 'Starting conversion :\nBasePath : ' + basePath + '\nTargetPath : ' + targetPath

# Will iterate all folders and one by one convert '*.pdb.gz' files to '*.mem'
# The path of .pdb.gz files is like "/usr/data/cvpr_shared/proteins/enzymes/EC2PDB/pdb/3/4/1/1/xxxx.pdb.gz"
# The output will be stored at "/usr/data/cvpr_shared/proteins/enzymes/EC2PDB/memmap_distmat/3/4/1/1/xxxx.mem"

# Some notes :
# 	1. pdb_id is sliced from the path of protein "..../3/4/1/1/[xxxx].pdb.gz" files . ie. path[-11:-7]
# 	2. using gzip.open(path_to_protein_pdb.gz_file) to pass in a file handle to retrieve size of structure
# 	3. There is no configuration based conversion yet. So all the folders in the basePath will be converted
# 	   once the script starts running. Make sure that the basePath contain strict directory structure.
#  Override folders

folders = [str(ec_class)]
pdb_id_shape = {}
pdb_count = 0
for ec_pos1 in folders:  # [3].4.1.1
    print 'EC pos1 : ' + ec_pos1 + '\n'
    pos_1_path = basePath + '/' + ec_pos1  # basePath/3
    ec_pos2_list = os.listdir(pos_1_path)  # 3.[4].1.1
    for ec_pos2 in ec_pos2_list:
        print 'EC pos2 : ' + ec_pos2 + '\n'
        pos_2_path = pos_1_path + '/' + ec_pos2  # basePath/3/4
        ec_pos3_list = os.listdir(pos_2_path)  # 3.4.[1].1
        for ec_pos3 in ec_pos3_list:
            print 'EC pos3 : ' + ec_pos3 + '\n'
            pos_3_path = pos_2_path + '/' + ec_pos3  # basePath/3/4/1
            ec = ('_').join([ec_pos1, ec_pos2, ec_pos3])
            ec_pos4_list = os.listdir(pos_3_path)  # 3.4.1.[1]
            for ec_pos4 in ec_pos4_list:
                print 'EC pos4 : ' + ec_pos4 + '\n'
                pos_4_path = pos_3_path + '/' + ec_pos4  # basePath/3/4/1/1
                proteins = os.listdir(pos_4_path)  # [3.4.1.1]/*.pdb
                for protein in proteins:
                    proteinpath = pos_4_path + '/' + protein
                    suffix = ec_pos1 + '/' + ec_pos2 + '/' + ec_pos3 + '/' + ec_pos4 + '/'
                    targetpath2 = targetPath + suffix + protein[:4] + '.mem'
                    print targetpath2
                    memmap_tmp = []
                    handle = gzip.open(proteinpath)
                    pdb_id = proteinpath[-11:-7]
                    protsize = get_size(pdb_id, handle)

                    if protsize is not False:
                        pdb_count += 1
                        id_shape = (protsize, protsize, nchannels)
                        pdb_id_shape[pdb_id] = (id_shape, ec)
                        if pdb_count%100 == 0:
                            print 'Protein count = {}'.format(pdb_count)

                    if not size_only:
                        if protsize is None or "1bav" in proteinpath:
                            append_pdb_data(pdb_id, proteinpath, message="Error in fetching size.")
                        # Create the memmaps and a list of all the memmaps that are being written
                        else:
                            tomemmap_lazy(read_fn=lambda memmap_tmp: get_dist_mat(pdb_id, proteinpath, memmap_tmp),
                                          memmap_filename=targetpath2, dtype=dtype, shape=(protsize, protsize, nchannels))

add_to_csv(pdb_id_shape)
print 'Conversion completed.'

