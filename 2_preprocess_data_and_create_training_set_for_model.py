"""
Script creates training set for a model. It takes as input raw data from Świgra parser (only correct sentences), transforms them to the format needed for a model and collect training examples on the specified schema.


Usage: 
	python   ................       -input_data_folder -output_data_folder

- input_data_folder - 
- output_data_folder - 

"""


import glob
import sys
import shutil
import os
import ntpath
import re
import numpy as np
import xml.etree.ElementTree as ET

from copy import deepcopy
from data_preprocessing import *



if __name__ == '__main__':


	np.random.seed(12345)
	
	if len(sys.argv)!=3:
		print(__doc__)
		sys.exit(0)
	
	input_data_folder = sys.argv[1]

	output_train_data_folder = sys.argv[2]


	for filename in np.array(os.listdir(input_data_folder)):


		name = ntpath.basename(filename)
		forest = ET.parse(os.path.join(input_data_folder,name))

		n_trees = number_of_trees_in_forest(forest)
		
		if n_trees == 1:
			trees = [get_positive_tree(forest)]
		elif n_trees <= 10:
			trees =  get_all_trees_randomly(forest)
		else:
			trees = [get_positive_tree(forest)]
			trees.extend(get_n_negative_trees_randomly(forest,10))
		
		for tree in trees:
			write_dependency_format(transform_to_dependency_format(tree), output_train_data_folder)



