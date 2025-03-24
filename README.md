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
3. Create .ipynb for graph creation
4. Rewrite get_attacks and use configs for attacks
5. Add MI-FGSM, APGD
6. Add ptlflow


