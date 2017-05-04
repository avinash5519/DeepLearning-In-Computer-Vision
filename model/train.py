# Deep Learning for Computer Vision practical course WS 2016/17
# Rajat Jain
# Protein function prediction from 2D representation of 3D structure

import datetime
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
split_criteria = 'strict'


# Different log files used during the complete training.

LOG_DIRECTORY = '/usr/prakt/w049/shared/logs'
IMAGE_DIRECTORY = '/usr/prakt/w049/shared/images'
MODEL_DIRECTORY = '/usr/prakt/w049/shared/model'
ERROR_FILE = '{}/{}_{}'.format(LOG_DIRECTORY, split_criteria, utils.time_stamped('error.log'))
LOSS_FILE = '{}/{}_{}'.format(LOG_DIRECTORY, split_criteria, utils.time_stamped('loss.log'))
LOG_FILE = '{}/{}_{}'.format(LOG_DIRECTORY, split_criteria, utils.time_stamped('train.log')) # Detailed logs
TRAINING_LOGS = '{}/{}_{}'.format(LOG_DIRECTORY, split_criteria, utils.time_stamped('training.log')) # Short and crisp
IMAGE_PATH = '{}/{}_'.format(IMAGE_DIRECTORY, split_criteria)
DISTANCE_MMAP_PATH = '/usr/data/cvpr_shared/proteins/enzymes/EC2PDB/distmat_mmap'

# Contains excel files which needs to precomputed when calculating distance matrices
# This is needed to get number of residues in a protein
PROTEIN_SIZE_DATA_PATH = '/usr/data/cvpr_shared/proteins/enzymes/EC2PDB/input_data'

# Time related functions and variables
current_milli_time = lambda: int(time.time() * 1000)

# The enzyme class to train and validate
ec_class = ['3']

# Parameters for the training
learning_rate = 0.00001
loops = 40 # epochs
load_previous_model = False # Do you require to initialize weights from a pre-trained model ?
model_file = '{}/4c_n_model_39.npz'.format(MODEL_DIRECTORY) # Change the model file_name here
is_send_mail = True # Enable/Disbale sending training report after every epoch on mail

'''
DO NOT EDIT anything after this line if you don't know what you are doing!
'''


def get_ec_tree(base_path, id_path_map, ec_class=['3', '4', '5', '6']):
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


def get_training_data(tree, split_type='naive', ec_class=['3','4','5','6']):
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
    train_data = []
    val_data = []
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
                            total = plist[:int(len(plist) * ratio)]
                            tr = total[:int(len(total) * ratio)]
                            train_data += tr
                            va = total[int(len(total) * ratio):]
                            val_data += va
                    elif split_type == 'strict':
                        list_level_4 = node_3._fpointer
                        nodes_total = list_level_4[:int(len(list_level_4) * ratio)]
                        nodes_tr = nodes_total[:int(len(nodes_total) * ratio)]
                        nodes_va = nodes_total[int(len(nodes_total) * ratio):]
                        for node in nodes_total:
                            plist = tree.get_node(node).data
                            if node in nodes_tr:
                                train_data += plist
                            elif node in nodes_va:
                                val_data += plist
    return train_data, val_data, target_classes


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
tree = get_ec_tree(DISTANCE_MMAP_PATH, id_path_map, ec_class)

# Get training , validation data and target clasees
logging.info("Creating data for network")
data_train, data_validation, target_classes = get_training_data(tree, split_criteria, ec_class)
logging.info("Training data initial length : {}".format(len(data_train)))
logging.info("Validation data initial length : {}".format(len(data_validation)))


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


def save_loss_data(type, epoch, loss):
    '''
    Write the information about loss for the given epoch to the file.
    LOSS_FILE = {LOG_DIRECTORY}/{split_criteria}_{timestamp}_loss.log
    :param type: Validation or Training loss
    :param epoch:
    :param loss:
    :return:
    '''
    with open(LOSS_FILE, "a") as loss_file:
        line_to_add = "{} # {} : {}\n".format(type, epoch, loss)
        loss_file.write(line_to_add)

# Prepare Theano variables for inputs and targets

input_var = T.tensor4('inputs')
target_var = T.matrix('targets')

# Create CNN model

logging.info("Building model and compiling functions...")
network = build_model(input_var)
prediction = lasagne.layers.get_output(network['fc7'], input_var)
loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
loss = loss.mean()

# Not calculating accuracy. Instead comparing outputs after the epoch is completed using confusion matrix
# If you still want to use, uncomment below line and make the necessary changes in the train function

# accuracy = T.mean(T.eq(T.argmax(prediction, axis=1), target_var), dtype=theano.config.floatX)
trainable_params = lasagne.layers.get_all_params(network['fc7'], trainable=True)

# For learning rate decay
# learning_rate = T.scalar(name='learning_rate')
updates = lasagne.updates.adam(loss, trainable_params, learning_rate=learning_rate)

train_fn = theano.function([input_var, target_var, learning_rate], [loss, T.argmax(prediction, axis=1), prediction, target_var],
                           updates=updates, allow_input_downcast=True)
validation_fn = theano.function([input_var, target_var],
                                [loss, T.argmax(prediction, axis=1), prediction, target_var],
                                allow_input_downcast=True)

## Start the onslaught!

logging.info("Starting training")

if load_previous_model:
    with np.load(model_file) as model:
        param_values = [model['arr_%d' % i] for i in range(len(model.files))]
    lasagne.layers.set_all_param_values(network['fc7'], param_values)
    logging.info("Loaded model !")


loss_history = []
# Skip ids which are causing memory mproblems or which raises exception while creation of protein,
# so that from the second epoch onwards we don't use these faulty ids.
ids_to_skip = []


def train(train_err, train_batches):
    for i in range(len(data_train)):
        try:
            pdbid = data_train[i]
            if pdbid in ids_to_skip:
                continue
            try:
                protein = Protein(pdbid, dataframe, id_path_map, target_classes)
                if protein.distance_mat_shape[0] > 700 or protein.distance_mat_shape[0] < 50: # Depends on GPU
                    logging.info("Skipping PDB : {} Size too large = {}".format(pdbid, protein.distance_mat_shape[0]))
                    ids_to_skip.append(pdbid)
                    continue
            except Exception as ex:
                msg = "Train# {}".format(ex)
                logging.info(msg)
                append_message(pdbid, ex)
                ids_to_skip.append(pdbid)
                continue
            targets = np.ndarray(shape=(1, len(target_classes)))
            targets[0][protein.label_idx] = 1
            inputs = protein.get_input_map()
            err, output_argmax, output_continuous, ya = train_fn(inputs, targets)
            logging.info("T# {} {} {}".format(ya, output_continuous, output_argmax))
            train_err += err
            train_batches += 1
            train_cfm[protein.label_idx][output_argmax[0]] += 1
        except Exception as ex:
            msg = "Train# Exception = {}".format(ex)
            logging.info(msg)
            append_message(pdbid, msg)
            pass
    return train_err, train_batches


def validate(validation_err, validation_batches):
    for j in range(len(data_validation)):
        try:
            pdbid = data_validation[j]
            if pdbid in ids_to_skip:
                continue
            try:
                protein = Protein(pdbid, dataframe, id_path_map, target_classes)
                if protein.distance_mat_shape[0] > 700 or protein.distance_mat_shape[0] < 50:
                    logging.info("Skipping PDB : {} Size too large = {}".format(pdbid, protein.distance_mat_shape[0]))
                    ids_to_skip.append(pdbid)
                    continue
            except Exception as ex:
                msg = "Val# {}".format(ex)
                logging.info(msg)
                append_message(pdbid, msg)
                ids_to_skip.append(pdbid)
                continue
            validation_targets = np.ndarray(shape=(1, len(target_classes)))
            validation_targets[0][protein.label_idx] = 1
            validation_inputs = protein.get_input_map()
            err, output_argmax, output_continuous, y_act = validation_fn(validation_inputs, validation_targets)
            logging.info(
                "Validation : Actual = {} , Predicted = {} , Approximated = {}".format(y_act[0], output_continuous[0], output_argmax[0]))
            validation_err += err
            validation_batches += 1
            validation_cfm[protein.label_idx][output_argmax[0]] += 1
        except Exception as ex:
            msg = "Val# Exception = {}".format(ex)
            logging.info(msg)
            append_message(pdbid, msg)
            pass
    return validation_err, validation_batches

try:
    for loop in range(loops):
        epoch = loop

        # Training #

        # Shuffle the input data
        random.shuffle(data_train)
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        train_loss = ''
        train_start_time = current_milli_time()

        # Log the target classes with their corresponding index
        for idx, val in enumerate(target_classes):
            logging.info("{} : {}".format(idx, val))
        logging.info("Running " + str(epoch))
        train_cfm = np.zeros(shape=(len(target_classes), len(target_classes)))
        train_err, train_batches = train(train_err, train_batches)
        train_end_time = current_milli_time()
        train_epoch_msg = "T# Epoch {} of {} took {:.3f}s".format(epoch + 1, loops, train_end_time - train_start_time)
        logging.info(train_epoch_msg)

        # Save the confusion matrix after every epoch
        logging.info("T# Confusion matrix :\n{}".format(train_cfm))
        image_path = '{}tr_epoch_{}.png'.format(IMAGE_PATH, epoch + 1)
        plot_confusion_matrix(train_cfm, target_classes, filename=image_path)

        # Calculate loss
        try:
            loss = train_err / train_batches
            loss_history.append(loss)
            train_loss = "  training loss:\t\t{:.6f}".format(loss)
            logging.info(train_loss)
            save_loss_data('TRAINING' , epoch, loss)
        except Exception as loss_ex:
            msg = "Train# Divide by zero : {}".format(loss_ex)
            logging.info(msg)
            append_message('', msg)
            continue

        # Validation #

        validation_err = 0
        validation_batches = 0
        validation_loss = ''
        validation_epoch_msg = ''
        if epoch % 1 == 0: # Change here if you don't want to run after every epoch
            validation_start_time = current_milli_time()
            validation_cfm = np.zeros(shape=(len(target_classes), len(target_classes)))
            validation_err, validation_batches = validate(validation_err, validation_batches)
            validation_end_time = current_milli_time()
            validation_epoch_msg = "V# Epoch {} of {} took {:.3f}s".format(epoch + 1, loops, time.time() - train_end_time)
            logging.info(validation_epoch_msg)

            # Save the validation confusion matrix
            logging.info("V# Confusion matrix :\n{}".format(validation_cfm))
            image_path = '{}val_epoch_{}.png'.format(IMAGE_PATH, epoch + 1)
            plot_confusion_matrix(validation_cfm, target_classes, filename=image_path)

            # Calculate validation loss
            try:
                val_loss = validation_err / validation_batches
                validation_loss = "  validation loss:\t\t{:.6f}".format(val_loss)
                logging.info(validation_loss)
                save_loss_data('VAL', epoch, val_loss)
            except Exception as loss_ex:
                msg = "Val# Divide by zero : {}".format(loss_ex)
                logging.info(msg)
                append_message('', msg)
                continue

        # Log the training information from the epoch to a file
        with open(TRAINING_LOGS, "a") as training_data:
            training_data.write(train_epoch_msg + '\n')
            training_data.write(train_loss + '\n')
            if epoch % 1 == 0:
                training_data.write(validation_epoch_msg + '\n')
                training_data.write(validation_loss + '\n')

        # Save the parameters
        if epoch % 1 == 0:
            model_path = '{}/{}_model_{}.npz'.format(MODEL_DIRECTORY, split_criteria, epoch + 1)
            np.savez(model_path,*lasagne.layers.get_all_param_values(network['fc7']))
            # Send mail if needed
            if is_send_mail:
                body = '{} {}\nAnalytics : \nT# {} \n V# {}\nSaved model : {}'.format(train_epoch_msg, validation_epoch_msg, train_loss, validation_loss, model_path)
                subject = "Training Report :
                    {}".format(utils.time_stamped('model'))
                utils.send_mail(body, subject=subject)
except KeyboardInterrupt:
    if is_send_mail:
        utils.send_mail('Somebody interrupted the training. Please check',
                        subject = 'INTERRUPTED Training')

# Uncomment if you want to save the loss history to a csv file
# import csv
# with open('{csv_file_name}', 'wb') as csv_file:
#     wr = csv.writer(csv_file)
#     wr.writerow(loss_history)

# Save the final model
np.savez('{}/{}_model_full.npz'.format(MODEL_DIRECTORY, split_criteria), *lasagne.layers.get_all_param_values(network['fc7']))
