## Modeling tips from the NVIDIA blog (by Kaggle competition winners)
1. Always use cross validation of data.
1. Make sure your training and validation sets have similar feature distributions.
1. Try many baseline models to start (MLP, RF, Linear, etc)
1. Engineer new features
1. Use ensemble methods: start wit your strongest baseline model and add new models with varied weights.
1. Stacking: train a second model on the first model's outputs or residuals.
1. Create synethetically labelled data with your best model to augment the dataset
1. Once you have the final model:
    * Train on all data
    * try different random seeds.  
