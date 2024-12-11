# Flow attack framework
## Adding new models
1. Add model to models/models
2. Add weights to _pretrained_weights
3. Add model loading in model_utils.py to load_and_import(). Make sure it takes images in [0,255] or [0,1]
4. Add model inference to compute_flow()
5. Add model as command line argument (#TODO)

## TODOS
1. <s>Finish model loading (add yaml file loading with arguments to import and load)</s>
2. <s>Add flow viz with flow_library</s>
3. Test model loading with datasets
4. Add MeFlow
5. Move MemFlow weights to _pretrained_weights folder
6. Add logging with mlflow
7. Create baseattack class
8. Add flow_benchmark models
9. Create argparse