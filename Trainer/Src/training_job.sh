#!/bin/bash

#job name
#SBATCH --job-name=identifier_training_job

#account for job 
#SBATCH --account=PHS0364

#when to be emailed 
#SBATCH --mail-type=ALL

#maximum amount of time
#SBATCH --time=1:30:00

#request number of cores 
#SBATCH --ntasks=6

#request number of GPUs per core 
#SBATCH --gpus-per-node=1

#enable modules that are needed for tensorflow 
module load python/3.7-2019.10
module load cuda/11.2.2
source activate tensor

#copy all files to the tempdirectory for faster I/O operations
cp $HOME/Identifiers/Gen1-4.11/identifierPackage.zip $TMPDIR
cd $TMPDIR
unzip -qq identifierPackage.zip

#run script 
cd $TMPDIR/PokemonIdentifier/Src/
python3 PokemonIdentifier.py

#copy log files from TMP directory back to home
cp $TMPDIR/PokemonIdentifier/Src/Logs.zip $HOME
