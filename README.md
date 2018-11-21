# ABLR_code
The source code of the paper: "To Find Where You Talk: Temporal Sentence Localization in Video with Attention Based Location Regression"


######################################################################################


./src/data_preparation_ActivityNet: Prepare data for ActivityNet Captions dataset

./src/data_preparation_Tacos: Prepare data for TACOS dataset

./src/model: ABLR model source file

./data: Folder for data preparation

#######################################################################################


External data to download or acquire: 

C3D video features for ActivityNet: http://activity-net.org/challenges/2016/download.html

TACOS dataset: http://www.coli.uni-saarland.de/projects/smile/page.php?id=tacos

ActicityNet Captions dataset: https://cs.stanford.edu/people/ranjaykrishna/densevid/

Glove word features: https://nlp.stanford.edu/projects/glove/ 
We use the word features "glove.840B.300d_dict.npy" when implement the ABLR model
