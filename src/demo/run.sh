#!/bin/sh

find . -name '.DS_Store' -type f -delete
cd demo/data_extract/
export CLASSPATH=../Packages/stanford-corenlp-full-2018-10-05/stanford-corenlp-3.9.2.jar
echo "Please tokenize this text." | java edu.stanford.nlp.process.PTBTokenizer
python3 make_datafiles.py ../demo_data ../demo_data
cd finished_files/
tar -xf val.tar
cd ../../fast_abs_rl-master/
export DATA=../data_extract/finished_files/
python3 decode_full_model.py --path=output/ --model=pretrained/model/ --beam=10 --val
