# Deep Learning for Computer Vision practical course WS 2016/17
# Avinash Kumar / Rajat Jain
# Protein function prediction from 2D representation of 3D structure

import logging
import os
import random
import time

import lasagne
import matplotlib
import numpy as np
import pandas as pd
import theano.tensor as T
from theano.tensor import *
from treelib import Tree

from protein import Protein

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools
import dlcv.utils.utility as utils


'''
These are the changeable parameters in the complete file.
'''

# How to split the training and validation set.
# Two possible options : 'naive' , 'strict'
split_criteria = 'naive'


# Different log files used during the complete training.

LOG_DIRECTORY = '/usr/prakt/w049/shared/logs'
IMAGE_DIRECTORY = '/usr/prakt/w049/shared/images'
MODEL_DIRECTORY = '/usr/prakt/w049/shared/model'
ERROR_FILE = '{}/{}_{}'.format(LOG_DIRECTORY, split_criteria, utils.time_stamped('error.log'))
LOSS_FILE = '{}/{}_{}'.format(LOG_DIRECTORY, split_criteria, utils.time_stamped('loss.log'))
LOG_FILE = '{}/{}_{}'.format(LOG_DIRECTORY, split_criteria, utils.time_stamped('test.log'))
TESTING_LOGS = '{}/{}_{}'.format(LOG_DIRECTORY, split_criteria, utils.time_stamped('testing.log'))
IMAGE_PATH = '{}/{}_'.format(IMAGE_DIRECTORY, split_criteria)
DISTANCE_MMAP_PATH = '/usr/data/cvpr_shared/proteins/enzymes/EC2PDB/distmat_mmap'

# Contains excel files which needs to precomputed when calculating distance matrices
# This is needed to get number of residues in a protein
PROTEIN_SIZE_DATA_PATH = '/usr/data/cvpr_shared/proteins/enzymes/EC2PDB/input_data/'

# Time related functions and variables
current_milli_time = lambda: int(time.time() * 1000)

# The enzyme class to train and validate
ec_class = ['3']

# Parameters for the training
model_file = '{}/4c_n_model_39.npz'.format(MODEL_DIRECTORY) # Change the model file_name here
is_send_mail = False # Enable/Disbale sending training report after every epoch on mail

'''
DO NOT EDIT anything after this line if you don't know what you are doing!
'''



def get_pdb_tree(base_path, id_path_map, ec_class=['3','4','5','6']):
    '''
    Iterate over base_path and construct a tree structure for all the enzyme classes
    Add 'data = protein_list' on the leaf node i.e. level = 4

    :param base_path: path which contains all the mem maps files starting with root level
    :param id_path_map: fills in this map of protein ids with their corresponding mmap file location
    :param ec_class:
    :return: tree object for the given ec_classes
    '''
    tree = Tree()
    tree.create_node("PDB", 0)  # root node
    folders = os.listdir(base_path)
    for ec_pos1 in folders:  # [3].4.1.1
        pos_1_path = base_path + '/' + ec_pos1  # base_path/3
        if ec_pos1 not in ec_class:
            logging.info("skipping folder : " + pos_1_path)
            continue
        tree.create_node(ec_pos1, ec_pos1, parent=0)
        logging.info('EC pos1 : ' + ec_pos1)
        ec_pos2_list = os.listdir(pos_1_path)  # 3.[4].1.1
        for ec_pos2 in ec_pos2_list:
            l2_ident = ec_pos1 + '_' + ec_pos2
            tree.create_node(ec_pos2, l2_ident, parent=ec_pos1)
            logging.info('EC pos2 : ' + ec_pos2)
            pos_2_path = pos_1_path + '/' + ec_pos2  # base_path/3/4
            ec_pos3_list = os.listdir(pos_2_path)  # 3.4.[1].1
            for ec_pos3 in ec_pos3_list:
                l3_ident = l2_ident + '_' + ec_pos3
                tree.create_node(ec_pos3, l3_ident, parent=l2_ident)
                logging.info('EC pos3 : ' + ec_pos3)
                pos_3_path = pos_2_path + '/' + ec_pos3  # base_path/3/4/21
                ec_pos4_list = os.listdir(pos_3_path)  # 3.4.21.[1]
                for ec_pos4 in ec_pos4_list:
                    l4_ident = l3_ident + '_' + ec_pos4
                    protein_list = []
                    tree.create_node(ec_pos3, l4_ident, parent=l3_ident, data=protein_list)
                    logging.info('EC pos4 : ' + ec_pos4)
                    pos_4_path = pos_3_path + '/' + ec_pos4  # base_path/3/4/21/1
                    mem_maps = os.listdir(pos_4_path)  # [3.4.21.1]/*.mem
                    for map in mem_maps:
                        protein_list.append(map[-8:-4])
                        protein_mmap_path = pos_4_path + '/' + map
                        id_path_map[protein_mmap_path[-8:-4]] = protein_mmap_path
    return tree


def get_test_data(tree, split_type='naive', ec_class=['3','4','5','6']):
    '''
    Divides data based on the type of split criteria provided. Defaults to 'naive'
    Uses data only for the provided ec_class list. Defaults to all available data.

    Makes use of information contained in the provided tree to iterate and split the data accordingly.

    :param tree: Tree object should contain all the information needed to split the data.
     Make sure to add corresponding proteins to the leaf nodes
    :param split_type: 'naive' or 'strict'
    :param ec_class: list of string eg. ['3']
    :return: training , validation set and target classes
    '''
    ratio = 0.7
    test_data = []
    children = tree.children(0)
    target_classes = []
    for node_1 in children:
        if node_1.identifier in ec_class:
            l1_children = tree.children(node_1.identifier)
            for node_2 in l1_children:
                l2_children = tree.children(node_2.identifier)
                for node_3 in l2_children:
                    target_classes.append(node_3.identifier)
                    l3_children = tree.children(node_3.identifier)
                    if split_type == 'naive':
                        for node_4 in l3_children:
                            plist = node_4.data
                            total_test_data = plist[int(len(plist) * ratio):]
                            test_data += total_test_data
                    elif split_type == 'strict':
                        list_level_4 = node_3._fpointer
                        nodes_total = list_level_4[int(len(list_level_4) * ratio):]
                        for node in nodes_total:
                            plist = tree.get_node(node).data
                            test_data += plist
    return test_data, target_classes


def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, filename=''):
    '''
    This function prints and plots and saves the confusion matrix.

    :param cm: confusion matrix to plot
    :param classes: target classes corresponding to entries in the confusion matrix
    :param title:
    :param cmap: color mapping of the plot
    :param filename: image filename to save the plot to
    :return:
    '''
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    if len(classes) < 10:
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(filename)
    plt.clf()


## Intialization part

# Initialize logger
logging.basicConfig(filename=LOG_FILE, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)

# Initialize pdbid vs mem file path anad label dictionary
id_path_map ={}
tree = get_pdb_tree(DISTANCE_MMAP_PATH, id_path_map, ec_class)

# Get training , validation data and target clasees
logging.info("Creating data for network")
data_test, target_classes = get_test_data(tree, split_criteria, ec_class)
logging.info("Testing data initial length : {}".format(len(data_test)))

# Initialize data frames containing information about number of residues in a protein
# Append different data frames together. Essentially you should have one excel file for each root level class.
frames = []
for class_id in ec_class:
    df = pd.read_excel('{}dict_{}.xlsx'.format(PROTEIN_SIZE_DATA_PATH, class_id)) # dict_3.xlsx is created using the csv file generated from distcane matrix calculation
    frames.append(df)
dataframe = pd.concat(frames)


## Build network


from lasagne.layers import InputLayer, DenseLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import MaxPool2DLayer as PoolLayer


def print_shape(net, layer_name):
    '''
    Prints the information for the given layer.

    :param net: the network object list
    :param layer_name: name of the layer contained in the network list above
    :return:
    '''
    logging.info('{} : {}'.format(layer_name, lasagne.layers.get_output_shape(net[layer_name])))


def build_model(input_var=None):
    '''
    Build the network

    :param input_var:
    :return: network list which contains all the layers
    '''
    net = {}
    net['input'] = InputLayer((None, 212, None, None), input_var=input_var)
    print_shape(net, 'input')
    net['conv1_1'] = ConvLayer(net['input'], num_filters=64, filter_size=3, nonlinearity=lasagne.nonlinearities.rectify,
                               W=lasagne.init.Normal())
    print_shape(net, 'conv1_1')
    net['conv1_2'] = ConvLayer(net['conv1_1'], num_filters=64, filter_size=3,
                               nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.Normal())
    print_shape(net, 'conv1_2')
    net['pool1'] = PoolLayer(net['conv1_2'], 2)
    print_shape(net, 'pool1')
    net['conv2_1'] = ConvLayer(net['pool1'], num_filters=128, filter_size=3,
                               nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.Normal())
    print_shape(net, 'conv2_1')
    net['conv2_2'] = ConvLayer(net['conv2_1'], num_filters=128, filter_size=3,
                               nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.Normal())
    print_shape(net, 'conv2_2')
    net['pool2'] = PoolLayer(net['conv2_2'], 2)
    print_shape(net, 'pool2')
    net['conv3_1'] = ConvLayer(net['pool2'], num_filters=256, filter_size=3,
                               nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.Normal())
    print_shape(net, 'conv3_1')
    net['conv3_2'] = ConvLayer(net['conv3_1'], num_filters=256, filter_size=3,
                               nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.Normal())
    print_shape(net, 'conv3_2')
    net['pool4'] = lasagne.layers.GlobalPoolLayer(net['conv3_2'], pool_function=T.max)
    print_shape(net, 'pool4')
    net['fc6'] = DenseLayer(net['pool4'], num_units=512, num_leading_axes=1,
                            nonlinearity=lasagne.nonlinearities.rectify)
    print_shape(net, 'fc6')
    net['fc7'] = DenseLayer(net['fc6'], num_units=len(target_classes), num_leading_axes=1, nonlinearity=lasagne.nonlinearities.softmax)
    print_shape(net, 'fc7')
    logging.info(lasagne.layers.count_params(net['fc7']))
    return net



def append_message(pdb_id, message):
    '''
    Append the error message to the error file with the corresponding pdb_id
    ERROR_FILE = {LOG_DIRECTORY}/{split_criteria}_{timestamp}_error.log
    :param pdb_id:
    :param message:
    :return:
    '''
    with open(ERROR_FILE, "a") as error_file:
        line_to_add = "{} : {} \n".format(pdb_id, message)
        error_file.write(line_to_add)


# Prepare Theano variables for inputs and targets

input_var = T.tensor4('inputs')
target_var = T.matrix('targets')

# Create CNN model

logging.info("Building model and compiling functions...")
network = build_model(input_var)
prediction = lasagne.layers.get_output(network['fc7'], input_var)
loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
loss = loss.mean()

# accuracy = T.mean(T.eq(T.argmax(prediction, axis=1), target_var),
#                   dtype=theano.config.floatX)
test_fn = theano.function([input_var, target_var] , [loss, T.argmax(prediction, axis=1), prediction, target_var], allow_input_downcast=True)


## Start the testing!

logging.info("Starting testing")

with np.load(model_file) as model:
    param_values = [model['arr_%d' % i] for i in range(len(model.files))]
lasagne.layers.set_all_param_values(network['fc7'], param_values)
logging.info("Loaded model !")


with np.load('/usr/prakt/w049/shared/feature/pmodel_20.npz') as f:
    param_values = [f['arr_%d' % i] for i in range(len(f.files))]
lasagne.layers.set_all_param_values(network['fc7'], param_values)
print "Loaded model !"

loss_history = []
ids_to_skip = []

# Shuffle the input data
random.shuffle(data_test)
# In each epoch, we do a full pass over the training data:
test_err = 0
test_batches = 0
test_cfm = np.zeros(shape=(len(target_classes), len(target_classes)))
test_start_time = current_milli_time()

# Log the target classes with their corresponding index
for idx, val in enumerate(target_classes):
    logging.info("{} : {}".format(idx, val))

for i in range(len(data_test)):
    try:
        pdbid = data_test[i]
        if pdbid in ids_to_skip:
            continue
        try:
            protein = Protein(pdbid, dataframe, id_path_map, target_classes)
            if protein.distance_mat_shape[0] > 700 or protein.distance_mat_shape[0] < 50:
                logging.info("Skipping PDB : {} Size too large = {}".format(pdbid, protein.distance_mat_shape[0]))
                ids_to_skip.append(pdbid)
                continue
        except Exception as ex:
            msg = "Test# {}".format(ex)
            logging.info(msg)
            append_message(pdbid, ex)
            ids_to_skip.append(pdbid)
            continue
        targets = np.ndarray(shape=(1, len(target_classes)))
        targets[0][protein.label_idx] = 1
        inputs = protein.get_input_map()
        err, yp, pred, ya = test_fn(inputs, targets)
        logging.info("Test# {} {} {}".format(ya, pred, yp))
        test_err += err
        test_batches += 1
        test_cfm[protein.label_idx][yp[0]] += 1
    except Exception as ex:
        msg = "Train# Exception = {}".format(ex)
        print msg
        append_message(pdbid, msg)
        pass
test_end_time = current_milli_time()
test_epoch_msg = "Test# took {:.3f}s".format(test_end_time - test_start_time)
logging.info(test_epoch_msg)

# Save the confusion matrix after every epoch
logging.info("Test# Confusion matrix :\n{}".format(test_cfm))
image_path = '{}test_epoch_{}.png'.format(IMAGE_PATH, current_milli_time)
plot_confusion_matrix(test_cfm, target_classes, filename=image_path)

# Calculate loss
try:
    t_loss = "  test loss:\t\t{:.6f}".format(test_err / test_batches)
    logging.info(t_loss)
except Exception as loss_ex:
    msg = "Test# Divide by zero : {}".format(loss_ex)
    logging.info(msg)
    append_message('', msg)

if is_send_mail:
    body = '{}\nAnalytics : \nTest# {}'.format(test_epoch_msg, t_loss)
    subject = "Testing Report : {}".format(utils.time_stamped('model'))
    utils.send_mail(body, subject=subject)

with open(TESTING_LOGS, "a") as test_data:
    test_data.write(t_loss + '\n')
