"""
Script collects files with correct sentences (sentences which were accepted by Swigra parser) and divide them into train and test sets.


Usage: 
	python 1_get_correct_forests_and_perform_train_test_split.py -input_data_folder -output_data_folder -test_size

- input_data_folder - folder with files (can be a folder of folders of files) being an output from Skladnica Parser.
- output_data_folder - folder to save correct sentences. In this folder two folders are created: "Train" and "Test"
- test_size - desired size of test set (if not specified, set to 2000)

"""


import glob
import sys
import shutil
import os
import ntpath
import re
import numpy as np
import xml.etree.ElementTree as ET

from data_preprocessing import is_forest_correct



if __name__ == '__main__':


	np.random.seed(12345)

	if len(sys.argv)==4:
		test_size = int(sys.argv[2]) 	
	elif len(sys.argv)==3:
		print("tests_size (3rd argument) not specified - 2000 is used")
		test_size = 2000
	elif len(sys.argv) != 3:
		print(__doc__)
		sys.exit(0)
	
	input_data_folder = os.path.join(sys.argv[1], "**","*.xml")

	output_train_data_folder = os.path.join(sys.argv[2], "Train")
	output_test_data_folder = os.path.join(sys.argv[2], "Test")

	os.makedirs(output_train_data_folder)
	os.makedirs(output_test_data_folder)


	N = 0
	print("Number of all sentences: ", len(list(glob.iglob(input_data_folder, recursive=True))))

	for filename in glob.iglob(input_data_folder, recursive=True):
		
		forest = ET.parse(filename)
		if is_forest_correct(forest):

			file_identifier = re.findall(sys.argv[1]+"(.+).xml", filename)[0].replace("/","__")

			shutil.copy2(filename, os.path.join(output_train_data_folder,file_identifier+".xml"))
			N = N + 1

	print("Number of correct sentences: ", N)


	random_sentences = np.random.choice(range(N), test_size, replace=False)
	for filename in np.array(os.listdir(output_train_data_folder))[random_sentences]:
		name = ntpath.basename(filename)
		shutil.move(os.path.join(output_train_data_folder,filename), os.path.join(output_test_data_folder,name))
		







