SHELL := bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c
.DELETE_ON_ERROR:
MAKEFLAGS += --warn-undefined-variables
MAKEFLAGS += --no-builtin-rules
.RECIPEPREFIX = >

data ?= data/example_data.tsv
lang ?= slo
chunks_column ?= date


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


DOCKER_DATA_ARGS := -v $$(realpath $(data)):/data.in

PREPROCESS_ARGS := --data_path /data.in \
                   --chunks_column $(chunks_column) \
                   --text_column text \
                   --lang $(lang) \
                   --output_dir /out \
                   --min_freq 10
out/.preprocessed.sentinel: out/.build.sentinel
> echo "Using data from $(data)"
> echo "Split is made over $(chunks_column) column"
> $(DOCKER) $(DOCKER_ARGS) ${DOCKER_DATA_ARGS} $(DOCKER_IMG) python /app/preprocess.py $(PREPROCESS_ARGS)
> touch $@
preprocess: out/.preprocessed.sentinel
.PHONY: preprocess

model_ref ?= EMBEDDIA/sloberta
num_train_epochs ?= 10

FINETUNE_ARGS := --train_file /out/train_lm.txt \
                 --validation_file /out/test_lm.txt \
                 --output_dir /out/models \
                 --data_path /data.in \
                 --chunks_column $(chunks_column) \
                 --model_name_or_path $(model_ref) \
                 --do_train true \
                 --do_eval true \
                 --per_device_train_batch_size 16 \
                 --per_device_eval_batch_size 4 \
                 --save_steps 20000 \
                 --evaluation_strategy steps \
                 --eval_steps 20000 \
                 --overwrite_cache \
                 --num_train_epochs $(num_train_epochs) \
                 --max_seq_length 512

finetune: out/train_lm.txt out/test_lm.txt
> echo "Using data from $(data)"
> echo "Split is made over $(chunks_column) column"
> echo "Training model $(model_ref) for $(num_train_epochs) epochs"
> $(DOCKER) $(DOCKER_ARGS) $(DOCKER_DATA_ARGS) $(DOCKER_IMG) python /app/finetune_mlm.py $(FINETUNE_ARGS)
.PHONY: finetune

EMBEDD_ARGS := --vocab_path /out/vocab.pickle \
               --embeddings_path /out/embeddings.pickle \
               --lang $(lang) \
               --path_to_fine_tuned_model $(model_ref) \
               --batch_size 16 \
               --max_sequence_length 256 \
               --device cuda

out/.embeddings.sentinel: out/vocab.pickle
> $(DOCKER) $(DOCKER_ARGS) $(DOCKER_IMG) python /app/get_embeddings_scalable.py $(EMBEDD_ARGS)
> touch $@
embeddings: out/.embeddings.sentinel
.PHONY: embeddings


metric ?= JSD
SEMANTIC_SHIFT_ARGS := --output_dir /out \
                       --embeddings_path /out/embeddings.pickle \
                       --random_state 123 \
                       --cluster_size_threshold 10 \
                       --metric $(metric)

out/.semantic-shift.sentinel: out/embeddings.pickle
> $(DOCKER) $(DOCKER_ARGS) $(DOCKER_IMG) python /app/measure_semantic_shift.py $(SEMANTIC_SHIFT_ARGS)
> touch $@
semantic-shift: out/.semantic-shift.sentinel
.PHONY: semantic-shift

target_words ?= "diplomat,objava"
INTERPRETATION_ARGS := --target_words $(target_words) \
                       --lang $(lang) \
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
