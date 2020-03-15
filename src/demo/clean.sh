#!/bin/sh

cd demo/data_extract/
rm -r cnn_stories_tokenized/ dm_stories_tokenized/ finished_files/
cd ../fast_abs_rl-master/output/
rm -r output/
rm log.json