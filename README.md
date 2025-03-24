# Flow attack framework
## Adding new models
1. Add model to models/models
2. Add weights to _pretrained_weights
3. Add model loading in model_utils.py to load_and_import(). Make sure it takes images in [0,255] or [0,1]
4. Add model inference to compute_flow()
5. Add model as command line argument (#TODO)

## TODOS
1. <s>Add saving iterations to FGSM, PGD</s>
2. <s>Run CosPGD, FGSM, PGD on full Kitti while saving iterations</s>
3. <s>Create .ipynb for graph creation</s>
4. Rewrite get_attacks and use configs for attacks
5. <s>Add MI-FGSM</s>, APGD
6. <s>Add ptlflow</s>

## TODO for ptlflow
1. <s>Add ptlflow to argparse</s>
2. <s>Add ptlflow functionality to other attacks</s>
3. *Decide checkpoints for each model* - Choosing for now checkpoint equal to dataset on which attack is run
4. Check why is the performance through ptlflow different to regular attack