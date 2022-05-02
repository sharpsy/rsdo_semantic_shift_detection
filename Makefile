SHELL := bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c
.DELETE_ON_ERROR:
MAKEFLAGS += --warn-undefined-variables
MAKEFLAGS += --no-builtin-rules
.RECIPEPREFIX = >

include config.mk

DATA ?= data/example_data.tsv
DATA_GROUP_COLUMN ?= date
DATA_LANG ?= slo
FINETUNE_MODEL_SOURCE ?= EMBEDDIA/sloberta
FINETUNE_TRAIN_EPOCHS ?= 10
EMBEDD_MODEL_SOURCE ?= out/models/
EMBEDD_METRIC ?= JSD
TARGET_WORDS ?= "diplomat,objava"


DOCKER := docker run
DOCKER_IMG := semantic-shift
DOCKER_ARGS := --gpus all \
               --rm \
               --env PYTHONUNBUFFERED=1 \
               --env TRANSFORMERS_CACHE=/out/.cache \
               -v $$(realpath out):/out

# Default - top level rule is what gets ran when you run just `make`
out/.build.sentinel: $(shell find src -type f) Dockerfile
> docker build . --tag=$(DOCKER_IMG)
> mkdir -p out
> touch $@
build: out/.build.sentinel
.PHONY: build


DOCKER_DATA_ARGS := -v $$(realpath $(DATA)):/data.in

PREPROCESS_ARGS := --data_path /data.in \
                   --chunks_column $(DATA_GROUP_COLUMN) \
                   --text_column text \
                   --lang $(DATA_LANG) \
                   --output_dir /out \
                   --min_freq 10
out/.preprocessed.sentinel: out/.build.sentinel
> echo "Using data from $(DATA)"
> echo "Split is made over $(DATA_GROUP_COLUMN) column"
> $(DOCKER) $(DOCKER_ARGS) ${DOCKER_DATA_ARGS} $(DOCKER_IMG) python /app/preprocess.py $(PREPROCESS_ARGS)
> touch $@
preprocess: out/.preprocessed.sentinel
.PHONY: preprocess


FINETUNE_ARGS := --train_file /out/train_lm.txt \
                 --validation_file /out/test_lm.txt \
                 --output_dir /out/models \
                 --data_path /data.in \
                 --chunks_column $(DATA_GROUP_COLUMN) \
                 --model_name_or_path $(FINETUNE_MODEL_SOURCE) \
                 --do_train true \
                 --do_eval true \
                 --per_device_train_batch_size 16 \
                 --per_device_eval_batch_size 4 \
                 --save_steps 20000 \
                 --evaluation_strategy steps \
                 --eval_steps 20000 \
                 --overwrite_cache \
                 --num_train_epochs $(FINETUNE_TRAIN_EPOCHS) \
                 --max_seq_length 512

finetune: out/train_lm.txt out/test_lm.txt
> echo "Using data from $(DATA)"
> echo "Split is made over $(DATA_GROUP_COLUMN) column"
> echo "Training model $(FINETUNE_MODEL_SOURCE) for $(FINETUNE_TRAIN_EPOCHS) epochs"
> $(DOCKER) $(DOCKER_ARGS) $(DOCKER_DATA_ARGS) $(DOCKER_IMG) python /app/finetune_mlm.py $(FINETUNE_ARGS)
.PHONY: finetune

EMBEDD_ARGS := --vocab_path /out/vocab.pickle \
               --embeddings_path /out/embeddings.pickle \
               --lang $(DATA_LANG) \
               --path_to_fine_tuned_model $(EMBEDD_MODEL_SOURCE) \
               --batch_size 16 \
               --max_sequence_length 256 \
               --device cuda

out/.embeddings.sentinel: out/vocab.pickle
> $(DOCKER) $(DOCKER_ARGS) $(DOCKER_IMG) python /app/get_embeddings_scalable.py $(EMBEDD_ARGS)
> touch $@
embeddings: out/.embeddings.sentinel
.PHONY: embeddings


SEMANTIC_SHIFT_ARGS := --output_dir /out \
                       --embeddings_path /out/embeddings.pickle \
                       --random_state 123 \
                       --cluster_size_threshold 10 \
                       --metric $(EMBEDD_METRIC)

out/.semantic-shift.sentinel: out/embeddings.pickle
> $(DOCKER) $(DOCKER_ARGS) $(DOCKER_IMG) python /app/measure_semantic_shift.py $(SEMANTIC_SHIFT_ARGS)
> touch $@
semantic-shift: out/.semantic-shift.sentinel
.PHONY: semantic-shift

INTERPRETATION_ARGS := --target_words $(TARGET_WORDS) \
                       --lang $(DATA_LANG) \
                       --input_dir /out \
                       --results_dir /out/results \
                       --cluster_size_threshold 10 \
                       --max_df 0.8 \
                       --num_keywords 10
interpretation:
> $(DOCKER) $(DOCKER_ARGS) $(DOCKER_IMG) python /app/interpretation.py $(INTERPRETATION_ARGS)
.PHONY: interpretation

clean:
> rm -rf out
.PHONY: clean


out/vocab.pickle: out/.preprocessed.sentinel
out/train_lm.txt: out/.preprocessed.sentinel
out/test_lm.txt: out/.preprocessed.sentinel
out/vocab_list_of_words.csv: out/.preprocessed.sentinel
/out/embeddings/embeddings.pickle: out/.embeddings.sentinel
