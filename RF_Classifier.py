#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, cohen_kappa_score, confusion_matrix
###############################################

###### Accuracy assesment for 4 classes: 'Other', 'TRN', 'SOJ', 'MIS'
def printMeasures(y_pred, y_test, verbose=True):
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average=None)
    kappa = cohen_kappa_score(y_test, y_pred)

    print(f"Overall Accuracy={(100*accuracy):.3f}%, Kappa={kappa:.4f}",
          f"F1={f1.mean()*100:.3f} (Other: {f1[0]*100:.3f}, TRN: {f1[1]*100:.3f}, SOJ: {f1[2]*100:.3f}, MIS: {f1[3]*100:.3f})")
    
    if verbose:
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred)        
        print(f"Other: Precision={100*precision[0]:.3f}%, Recall={100*recall[0]:.3f}%")
        print(f"TRN: Precision={100*precision[1]:.3f}%, Recall={100*recall[1]:.3f}%")
        print(f"SOJ: Precision={100*precision[2]:.3f}%, Recall={100*recall[2]:.3f}%")
        print(f"MIS: Precision={100*precision[3]:.3f}%, Recall={100*recall[3]:.3f}%")
        
        print(f"Overall (average): Precision={100*precision.mean():.3f}%, Recall={100*recall.mean():.3f}%")

        class_names = ['Other', 'TRN', 'SOJ', 'MIS']
        cm = confusion_matrix(y_test, y_pred)  
        print("\nConfusion matrix:")
        print(f"[Other] [{cm[0,0]:5d}, {cm[0,1]:5d}, {cm[0,2]:5d}, {cm[0,3]:5d}]")
        print(f"[TRN] [{cm[1,0]:5d}, {cm[1,1]:5d}, {cm[1,2]:5d}, {cm[1,3]:5d}]")
        print(f"[SOJ] [{cm[2,0]:5d}, {cm[2,1]:5d}, {cm[2,2]:5d}, {cm[2,3]:5d}]")
        print(f"[MIS] [{cm[3,0]:5d}, {cm[3,1]:5d}, {cm[3,2]:5d}, {cm[3,3]:5d}]")
      
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=class_names, yticklabels=class_names)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')

#### Define year of train and test 
year_combinations = [(2021, 2020)] 

#### Define the traing and test arrayes for classification. The pairs below are the name of arrays in my dataset.
#######You can select the featuers of each array, for example using the three first featuers S2[:,:,(0,1,2)]

dataset_pairs = [
    ("S2", "S2"),  # Pair 1
   # ("S1", "S1"),      # Pair 2
   # ("polari_indices", "polari_indices"),    # Pair 3
   # ("S1Harmonic_Coef", "S1Harmonic_Coef"),                # Pair 4
   # ("S2Harmonic_Coef", "S2Harmonic_Coef"),                # Pair 4
]
   
for year_train, year_test in year_combinations:
    model_name =  "RF"
    rng_seed =  42
    region_train = 'Tarbes'
    region_test = 'Dijon'
    name = 'S2_important' ### Replace by the name of your own dataset
    np.random.seed(rng_seed)             
        
    ##### Define the directory for the results
    in_directory_out = f'F:/Project/results/{region_train}{year_train}_{region_test}{year_test}/{model_name}'
    os.makedirs(in_directory_out, exist_ok=True)

    for train_name, test_name in dataset_pairs:
        ind = f'{train_name}'
        name_out=f'{model_name}_{name}_{ind}_{region_train}{year_train}_{region_test}{year_test}.txt'
        output_path = f"{in_directory_out}/{name_out}"
        sys.stdout = open(output_path, "w")
            
        # Load training data
        in_directory1 = f'F:/Project/{region_train}{year_train}/'
        train_data_path = f'{in_directory1}/{name}.npz'         
        
        dataset1 = np.load(train_data_path, allow_pickle=True)  
        array_names1 = dataset1.files
        print(array_names1)#### Print the name of arrays in npz file
            
        x_train = dataset1[train_name].astype(np.float64)

        # Load test data
        in_directory2 = f'F:/Project/{region_test}{year_test}/'
        test_data_path = f'{in_directory2}/{name}.npz'         
        dataset2 = np.load(test_data_path, allow_pickle=True)
        array_names2 = dataset2.files
        print(array_names2)
            
        x_test = dataset2[test_name].astype(np.float64)
                 
        print(x_train.shape)
        print(x_test.shape)
        
        #### Normalize the data
        x_train = (x_train - np.percentile(x_train, 1)) / (np.percentile(x_train, 99) - np.percentile(x_train, 1))
        x_train[x_train > 1] = 1
        x_train[x_train < 0] = 0

        x_test = (x_test - np.percentile(x_test, 1)) / (np.percentile(x_test, 99) - np.percentile(x_test, 1))
        x_test[x_test > 1] = 1
        x_test[x_test < 0] = 0
            
        #### Flatten the data from the shape for example (20,3,4) to (20,12)
        x_train_flattened = x_train.reshape(x_train.shape[0], -1)
        x_test_flattened = x_test.reshape(x_test.shape[0], -1)
                
        print(x_train_flattened.shape)
        print(x_test_flattened.shape)

        # Label encoding for 4 classes: 'Other', 'TRN', 'SOJ', 'MIS'
        label_to_idx = {"TRN": 1, "SOJ": 2, 'MIS': 3}
        y_train = np.array([label_to_idx.get(label, 0) for label in dataset1["y"]])
        y_test = np.array([label_to_idx.get(label, 0) for label in dataset2["y"]])

         # Train Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=100,  # Number of trees in the forest
            random_state=rng_seed)

        # Fit the model
        rf_model.fit(x_train_flattened, y_train)

        # Predict on the test set
        y_pred_test = rf_model.predict(x_test_flattened)


        print(f"\n=================================\n PERFORMANCE\n=================================\n")
        printMeasures(y_pred_test, y_test, verbose=True)      
        in_directory_fig = f'F:/Project/cm/{region_train}{year_train}_{region_test}{year_test}/{model_name}/'
        os.makedirs(in_directory_fig, exist_ok=True)
        cm_output_path = f'{in_directory_fig}/{model_name}_{name}_{ind}_{region_train}{year_train}_{region_test}{year_test}.png'
        plt.savefig(cm_output_path, bbox_inches='tight')
        plt.show()


# In[ ]:




