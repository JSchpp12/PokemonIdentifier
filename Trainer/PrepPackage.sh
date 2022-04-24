#!/bin/bash
rm datasetPackage.zip
rm srcPackage.zip
zip identifierPackage.zip -r ./Datasets/Main/ 
zip srcPackage.zip -r ./Src/