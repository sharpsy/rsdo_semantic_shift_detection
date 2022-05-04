# Path to the data file in TSV format. It should contain `text` column
# with the data that will be analyzed, a column that will be used to
# group data (`DATA_GROUP_COLUMN`) and can contain a number of other
# columns.
DATA = data/example_data.tsv

# Data column from `DATA` file. This column will be used to group the
# data before the analysis.
DATA_GROUP_COLUMN = date

# Language of the data. Preprocessing and language model choice will
# be made based on this value. Supported values are `slo` for
# Slovenian and `en` for English.
DATA_LANG = slo

# Reference to the model that will be finetuned. It can be a directory
# of trained model, or a reference to a huggingface model that will be
# downloaded.
FINETUNE_MODEL_SOURCE = EMBEDDIA/sloberta

# Number of epoch for model finetuning on the `DATA`.
FINETUNE_TRAIN_EPOCHS = 10

# Reference to the model that will be used in evaluating semantic
# shift. It can be a directory of trained model, or a reference to a
# huggingface model that will be downloaded.
EMBEDD_MODEL_SOURCE = out/models/

# Metric used in quantifying semantic change.
EMBEDD_METRIC = JSD

# Words used to analyse the semantic change
TARGET_WORDS = "diplomat,objava"

# Device used for CUDA enabled computation, cuda or cpu
DEVICE = cuda
