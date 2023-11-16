#!/bin/bash -l

module load unstable neurodamus-neocortex
export USE_NEURODAMUS=1


#ticket: https://bbpteam.epfl.ch/project/issues/browse/BBPP134-996


#config_file:
#/gpfs/bbp.cscs.ch/project/proj134/scratch/tickets/BBPP134-901-emodel-generalisation-example/zisis/stage/recipe.json

#morphs:
#/gpfs/bbp.cscs.ch/data/scratch/proj134/workflow-outputs/20102023-20dd0dc6-e206-45f2-9c18-78b2edeb7744/morphologyAssignmentConfig/build/morphologies

#nodes:
#/gpfs/bbp.cscs.ch/data/scratch/proj134/workflow-outputs/20102023-20dd0dc6-e206-45f2-9c18-78b2edeb7744/morphologyAssignmentConfig/build/nodes.h5

#output_dir:
#/gpfs/bbp.cscs.ch/data/scratch/proj134/tickets/BBPP134-901-emodel-generalisation-example/BBPP134-901-emodel-generalisation-example

rm -rf config local

emodel-generalisation -v assign \
    --input-node-path /gpfs/bbp.cscs.ch/data/scratch/proj134/workflow-outputs/20102023-20dd0dc6-e206-45f2-9c18-78b2edeb7744/morphologyAssignmentConfig/build/nodes.h5 \
    --config-path /gpfs/bbp.cscs.ch/project/proj134/scratch/tickets/BBPP134-901-emodel-generalisation-example/zisis/stage/recipe.json \
    --output-node-path assigned_nodes.h5


emodel-generalisation -v adapt \
    --input-node-path assigned_nodes.h5 \
    --output-csv-path adapt_df.csv \
    --output-node-path adapted_nodes.h5 \
    --morphology-path /gpfs/bbp.cscs.ch/data/scratch/proj134/workflow-outputs/20102023-20dd0dc6-e206-45f2-9c18-78b2edeb7744/morphologyAssignmentConfig/build/morphologies \
    --config-path /gpfs/bbp.cscs.ch/project/proj134/scratch/tickets/BBPP134-901-emodel-generalisation-example/zisis/stage/recipe.json \
    --hoc-path hoc \
    --parallel-lib multiprocessing
