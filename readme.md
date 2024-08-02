# Multi-view Transformer-based Network (MTN)

Source code for "Multi-view Transformer-based Network for Prerequisite Learning in Concept Graphs".

## Overview

The MTN model is implemented using [PyTorch](https://pytorch.org/) and [Transformers](https://huggingface.co/transformers/). It leverages a multi-view transformer-based network to learn prerequisites in concept graphs.

## Requirements

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Data Processing & Concept Graph Construction

The data processing and concept graph construction steps are implemented in the `concept_graph_construction.py` script. This script reads data from the specified directory and constructs the concept graph for each dataset.

To execute the script, use the following command:

```bash
python concept_graph_construction.py
```

## Usage

Follow the steps below to use MTN for training on various datasets and settings.

1. **Download the Benchmarks**

   Download the benchmarks from [this link](https://drive.google.com/file/d/1ufYxZG4HPIAMzW1bxeGTnrEM2w4xQOcn/view?usp=sharing) and extract them to the root directory (`./`).

2. **Train the Model**

   Train the model on various datasets and settings using the commands below:

   #### Conventional Setting

   Train on a single fold of the dataset in the conventional setting. The `fold_id` can be any integer from 0 to 4. Replace `$dataset_name$` with the appropriate dataset name and `$fold_id$` with the fold ID to run the following command:

    ```bash
    python main.py --data_path benchmarks/conventional_setting/$dataset_name$ --fold_id $fold_id$
    ```

   #### Hard Setting

   Similarly, in the hard setting, replace `$dataset_name$` and `$fold_id$` with the dataset name and fold ID (0-4) respectively:

    ```bash
    python main.py --data_path benchmarks/hard_setting/$dataset_name$ --fold_id $fold_id$
    ```
