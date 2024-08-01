# Essential gene detection

This repository implements a method for essential gene detection using Graph Isomorphism Networks (GINs) called EssentialGens.

Data used is available at: https://drive.google.com/file/d/18TyM7WvZe5QxGCEAxJUWH0kHmi2fWCxa/view?usp=sharing

Download the data and place the entire folter into data folder

### Requirements:

Python 3.x
Required libraries (can be installed using pip install -r requirements.txt)

### Running the Model:

To train and evaluate the model, execute the following command:

`python runners/run_essential_gin.py --train --n_runs 10 --organism <organism> --ppi <ppi_network> [--expression] [--sublocs] [--orthologs]`

### Arguments:

--train: Train the model (required for training and evaluation).

--n_runs: Number of training runs (default: 10).

--organism: Organism to analyze (options: human, coli, melanogaster, yeast).

--ppi: Protein-protein interaction network database (options: string, biogrid, dip).

--expression: Include gene expression data (optional).

--sublocs: Include subcellular localization data (optional).

--orthologs: Include orthology information (optional).

### Data Inclusion:

Specify --expression, --sublocs, or --orthologs to integrate the respective data source into the protein-protein interaction network.
Omit these arguments to utilize only the PPI network for training.

The pre-processing steps from the essential-gene-detection project (https://github.com/JSchapke/essential-gene-detection).
