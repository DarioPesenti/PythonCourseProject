### What is it
Custom 2 layer convolutional network used to make experiment on embeddings convexity.
Trained on 3dshapes dataset (https://github.com/google-deepmind/3d-shapes), a syntethic dataset containing 480000 images of shapes in every combination of floor colour, wall colour, object colour, scale, shape and orientation.

### Hyperparameters
Train batch_size=128, ADAM optimizer, lr=10e-4. We performed no hyperparameter search since the dataset is synthetic and extremely simple.

### Python and packages
Python 3.9, install the packages with **pip install -r requirements.txt**.

### What scripts you can ignore
-dset_setup.py creates a sliced dataset ex novo, you don't need to run it, it's already prepared. If you chose to, beware the script will take A LOT
-Model.py contains the model and embeddings classifier
-convexity.py contains convexity analyses functions (WIP). 

### What to run
Train.py is the script to launch to perform the training and analyses.
It allows 4 arguments (you can call python train.py -h to get info about them): 
-w: enables wandb logging for train/test accuracy and loss (default False) #NOTE: No matplotlib functions have been abused for this project, use this flag to see the pretty graphs
-e: number of training epochs (default 20, but it achieves max accuracy after at most 5)
-c: perform convexity analyses. The output is stil to be refined and the results not definitive at all (default False).
-s: random seed value. Default 42. 
If the GPUs are overworked, uncomment line 162 on train.py to choose the preferred device.

