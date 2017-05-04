# Deep Learning for Computer Vision practical course WS 2016/17
# Rajat Jain
# Protein function prediction from 2D representation of 3D structure

Project structure:

--model
    |
    --train.py
    --test.py
    --protein.py
--processor
    |
    --ec2pdb_downloader.py
    --memmap_distmat.py
    --memmap_torsion.py
--utils
    |
    --utility.py
    --logging.conf


Usage :

ec2pdb_downloader.py

1. Download pdb files from https://files.rcsb.org/download/<PDB_ID>.pdb
2. To download specific class use --ec_class command line argument
    eg. python ec2pdb_downloader.py --ec_class 1
        python ec2pdb_downloader.py --ec_class 1 2 4
3. This stores the files at '/usr/data/cvpr_shared/proteins/enzymes/EC2PDB/pdb/'

memmap_distmat.py

1. Use this to calculate distance matrices and store them as memmap file at '/usr/data/cvpr_shared/proteins/enzymes/EC2PDB/distmat_mmap/'
2. To compute for specific class use --ec_class command line argument
    eg. python memmap_distmat.py --ec_class 1
3. Also needed to create a dict_<ec_class>.csv file which contains data about all the proteins and corresponding size.
4. IMPORTANT : Convert this csv file to excel separating each value in different column. Also add headers 'pdbid', 'rows', 'cols', 'depth', 'label'
    This is used while training (Consider this as an important part of dataset)
5. If you need to only get this csv file for protein size , then use --size_only command line argument
    eg. python memmap_distmat.py --ec_class 1 --size_only

memmap_torsion.py

1. Use this to calculate torsion angles (phi, psi and 5 side chain angles) and store them as memmap file at '/usr/data/cvpr_shared/proteins/enzymes/EC2PDB/rotamers_mmap/'
2. To compute for specific class use --ec_class command line argument
    eg. python memmap_distmat.py --ec_class 1

train.py

1. All the important configurations are listed at top few lines of the file.
2. Provide the information about training and validation set split inside the file. Two possible values 'naive' or 'strict'
3. Usage : python train.py [defaults to class 3]

test.py

1. Provide the model to initialize the network inside the file.
2. Provide the information about split criteria to get test data inside the file. Two possible values 'naive' or 'strict'
3. Usage : python test.py [defaults to class 3]




List of packages in my environment (though you might not need all of them, and there are few like treelib which needs to be added):

(protein) w049@atcremers60:~$ conda list
# packages in environment at /usr/prakt/w049/anaconda2/envs/protein:
#
beautiful-soup            4.3.2                    py27_0
biopython                 1.68                np111py27_0
cairo                     1.12.18                       6
cycler                    0.10.0                   py27_0
dbus                      1.10.10                       0
expat                     2.1.0                         0
fontconfig                2.11.1                        6
freetype                  2.5.5                         1
glib                      2.43.0                        1
gst-plugins-base          1.8.0                         0
gstreamer                 1.8.0                         0
icu                       54.1                          0
jpeg                      8d                            2
libffi                    3.2.1                         0
libgcc                    5.2.0                         0
libgfortran               3.0.0                         1
libpng                    1.6.22                        0
libxcb                    1.12                          1
libxml2                   2.9.2                         0
matplotlib                1.5.3               np111py27_1
mkl                       11.3.3                        0
numpy                     1.11.2                   py27_0
openssl                   1.0.2j                        0
pandas                    0.19.2              np111py27_0
pip                       9.0.1                    py27_0
pixman                    0.32.6                        0
pycairo                   1.10.0                   py27_0
pyparsing                 2.1.4                    py27_0
pyqt                      5.6.0                    py27_0
python                    2.7.12                        1
python-dateutil           2.6.0                    py27_0
pytz                      2016.7                   py27_0
qt                        5.6.0                         1
readline                  6.2                           2
scikit-learn              0.18.1              np111py27_0
scipy                     0.18.1              np111py27_0
setuptools                27.2.0                   py27_0
sip                       4.18                     py27_0
six                       1.10.0                   py27_0
sqlite                    3.13.0                        0
tk                        8.5.18                        0
wheel                     0.29.0                   py27_0
xlrd                      1.0.0                    py27_0
zlib                      1.2.8                         3
