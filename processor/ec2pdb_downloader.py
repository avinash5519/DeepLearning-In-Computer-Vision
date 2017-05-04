# Deep Learning for Computer Vision practical course WS 2016/17
# Rajat Jain
# Protein function prediction from 2D representation of 3D structure

import os
import urllib
import sys
from bs4 import BeautifulSoup
import argparse

def get_path(key):
    pathList = key.split('.')
    path = ('/').join(pathList)
    basePath = '/usr/data/cvpr_shared/proteins/enzymes/EC2PDB/pdb/' + path
    return basePath


def download_protein(path, proteinFile):
    url = 'https://files.rcsb.org/download/' + proteinFile
    filename = path + '/' + proteinFile
    urllib.urlretrieve(url, filename)


def download_file(id_map):
    """ Iterate over all EC map and their corresponding set of pdb_ids and
    downloads the files in the EC subfolder"""
    counter = 0
    for key,value in id_map.iteritems():
        path = get_path(key)
        if not os.path.exists(path):
            os.makedirs(path)

        for pid in value:
            counter = counter + 1
            if(counter%50==0) :
                print "downloaded " + str(counter) + " files"
            proteinFile = pid + '.pdb.gz'
            download_protein(path, proteinFile)


def get_pdb_ids(ec_number):
    """Use HTML parser Beautiful Soup to look for 'a' tag and href attribute.
    If it contains "/pdbsum/xxxx" , add that id to a set for this EC """
    exists = True
    url_prefix = "https://www.ebi.ac.uk/thornton-srv/databases/cgi-bin/enzymes/GetPage.pl?ec_number="
    url = url_prefix + ec_number
    pdb_id_set = set()
    source = urllib.urlopen(url)
    parsed_html = BeautifulSoup(source)
    is_invalid = parsed_html.find_all(text="Invalid EC number: ")
    if len(is_invalid) is not 0:
        return (pdb_id_set,False)
    pdb_sum = parsed_html.find_all('a')
    for pdb_tag in pdb_sum:
        href_tag = pdb_tag.attrs
        val = href_tag["href"]
        if 'pdbsum' in val:
            val = val[-4:]
            pdb_id_set.add(val)
    return (pdb_id_set, True)


def ec_exists(suf):
    exists = False
    url_prefix = "https://www.ebi.ac.uk/thornton-srv/databases/cgi-bin/enzymes/GetPage.pl?ec_number="
    url = url_prefix + suf
    source = urllib.urlopen(url)
    parsed_html = BeautifulSoup(source)
    invalid = parsed_html.find_all(text="Invalid EC number: ")
    if len(invalid) is 0:
        exists = True
    return exists


def get_protein_ids(ec_class):
    """Iterates over all EC from 1.1.1.1 and search for "pdbsum" in href tag to
    find all pdb's for that specific class. Save all the ids in a set against the
    suffix say 1.1.1.1. Return the map of all suffix(EC) with their corresponding set of all pdb ids"""

    ec_pid_map = {}
    suffix_does_not_exist = set()
    for level_1 in ec_class:
        for level_2 in range(1,100):
            suf_2 = str(level_1) + '.' + str(level_2) + '.-.-'
            print "2nd : Checking : " + suf_2
            if ec_exists(suf_2) is False:
                print "Skipping : " + suf_2
                continue
            for level_3 in range(1,100):
                suf_3 = str(level_1) + '.' + str(level_2) + '.' + str(level_3) + '.-'
                print "3rd : Checking : " + suf_3
                if ec_exists(suf_3) is False:
                    print "Skipping : " + suf_3
                    continue
                for level_4 in range(1,500):
                    suffix = '{}.{}.{}.{}'.format(str(level_1), str(level_2), str(level_3), str(level_4))
                    print "4th Checking : " + suffix
                    pdb_id_set,is_exists = get_pdb_ids(suffix)
                    if is_exists is False:
                        print "No more valid EC's from class : " + suffix
                        break
                    if (len(pdb_id_set)):
                        ec_pid_map[suffix] = pdb_id_set
                    else:
                        suffix_does_not_exist.add(suffix)
    return ec_pid_map

def main(argv):
    desc = """Lookup and download pdb files for given ec_class"""
    arg = argparse.ArgumentParser(description=desc)
    arg.add_argument('--ec_class', metavar='N', type=int, choices=range(7), default=[3], nargs='+',
                   help='EC class to process')
    args = arg.parse_args()
    print "Getting PDB ids -"
    print args.ec_class
    id_map = get_protein_ids(args.ec_class)
    count = sum(len(v) for v in id_map.itervalues())
    print "Downloading files : " + str(count)
    download_file(id_map)

if __name__ == '__main__':

    main(sys.argv[1:])
