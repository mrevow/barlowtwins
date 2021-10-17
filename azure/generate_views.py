import pandas as pd
import numpy as np
import dateutil
import os


"""
data plit strategy from Team's chat:
Supervised
10k for training broken into 4 files each with distinct counts (1K, 2k, 3k, 4k, ) 
That way using combos we can create any training size in the range 1-10k
1k for validation
test set

unsupervised all remaining data
5k validation - remaining for train
"""

view_output = "C:/Source/Hack/trainConfigs"

all_fn = pd.read_csv(os.path.join(view_output, 'all.txt'), header=None, names=['file'])
unlabeled = pd.read_csv(os.path.join(view_output, "data_music-detection-unlabeled2.log"), header=None, sep="\t", names=['file'])
print(f"{len(all_fn)} files in all_fn and {len(unlabeled)} files in unlbeled -- {len(all_fn)+len(unlabeled)} in totla.")

all_fn['labels'] = all_fn['file'].apply(lambda i: i.split("_")[0])
print("\n*** distribution of labels in all.txt ***")
all_fn['labels'].value_counts(normalize=True)


# select four sets of files for supervised train and one set for supervied validation

all_fn_working = all_fn.copy()

supervised_train_1K = all_fn_working.sample(n=1000, random_state=5)
print(f"\n*** distribution of labels in supervised_train_1K.txt -- {len(supervised_train_1K)} ***")
print(supervised_train_1K['labels'].value_counts(normalize=True))

all_fn_working = all_fn_working[~all_fn_working.file.isin(supervised_train_1K.file)]
supervised_train_2K = all_fn_working.sample(n=2000, random_state=5)
print(f"\n*** distribution of labels in supervised_train_2K.txt -- {len(supervised_train_2K)} ***")
print(supervised_train_2K['labels'].value_counts(normalize=True))

all_fn_working = all_fn_working[~all_fn_working.file.isin(supervised_train_2K.file)]
supervised_train_3K = all_fn_working.sample(n=3000, random_state=5)
print(f"\n*** distribution of labels in supervised_train_3K.txt -- {len(supervised_train_3K)} ***")
print(supervised_train_3K['labels'].value_counts(normalize=True))

all_fn_working = all_fn_working[~all_fn_working.file.isin(supervised_train_3K.file)]
supervised_train_4K = all_fn_working.sample(n=4000, random_state=5)
print(f"\n*** distribution of labels in supervised_train_4K.txt -- {len(supervised_train_4K)} ***")
print(supervised_train_4K['labels'].value_counts(normalize=True))

all_fn_working = all_fn_working[~all_fn_working.file.isin(supervised_train_4K.file)]
supervised_validation_1K = all_fn_working.sample(n=1000, random_state=5)
print(f"\n*** distribution of labels in supervised_validation_1K.txt -- {len(supervised_validation_1K)} ***")
print(supervised_validation_1K['labels'].value_counts(normalize=True))

# keep the remaining for unsupervised train
all_fn_working = all_fn_working[~all_fn_working.file.isin(supervised_validation_1K.file)]
print(f"\nremaining files in all.txt: {len(all_fn_working)} that will go to unsupervised train, with distribution:")
print(all_fn_working['labels'].value_counts(normalize=True))


# select two sets of files for unsupervised train/validation

unlabeled_working = unlabeled.copy()

unsupervised_valiation_5K = unlabeled_working.sample(n=5000, random_state=5)
unsupervised_train = unlabeled_working[~unlabeled_working.file.isin(unsupervised_valiation_5K.file)]
print(f"unlabeled file split into validation {len(unsupervised_valiation_5K)} and train: {len(unsupervised_train)}")


print(f"*** saving the view files in {view_output} ***")
supervised_train_1K['file'].to_csv(os.path.join(view_output, 'supervised_train_1K.txt'), header=None, index=False)
supervised_train_2K['file'].to_csv(os.path.join(view_output, 'supervised_train_2K.txt'), header=None, index=False)
supervised_train_3K['file'].to_csv(os.path.join(view_output, 'supervised_train_3K.txt'), header=None, index=False)
supervised_train_4K['file'].to_csv(os.path.join(view_output, 'supervised_train_4K.txt'), header=None, index=False)
supervised_validation_1K['file'].to_csv(os.path.join(view_output, 'supervised_validation_1K.txt'), header=None, index=False)
all_fn_working['file'].to_csv(os.path.join(view_output, 'unsupervised_train_128K_fromAllTxt.txt'), header=None, index=False)

unsupervised_valiation_5K['file'].to_csv(os.path.join(view_output, 'unsupervised_valiation_5K_fromUnlabeled2.txt'), header=None, index=False)
unsupervised_train['file'].to_csv(os.path.join(view_output, 'unsupervised_train_1_8M_fromUnlabeled2.txt'), header=None, index=False)

# checking for overlaps
y = [x for x in list(supervised_validation_1K['file']) if x in list(supervised_train_4K['file'])]
print(f"files in supervised_validation_1K that are also in supervised_train_1K {len(y)}")