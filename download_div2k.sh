#!/bin/bash
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X2.zip
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X2.zip
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip

mkdir div2k
unzip -q DIV2K_valid_LR_bicubic_X2.zip -d div2k
unzip -q DIV2K_train_LR_bicubic_X2.zip -d div2k
unzip -q DIV2K_train_HR.zip -d div2k
unzip -q DIV2K_valid_HR.zip -d div2k
