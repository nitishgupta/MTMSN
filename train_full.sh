SEED=10
  

# SERIALIZATION_DIR=./resources/semqa/checkpoints/drop-MTMSN/full_data/S_${SEED}/MTMSNModel

SERIALIZATION_DIR=/shared/nitishg/checkpoints/drop/MTMSN/full_data/S_${SEED}/MTMSNModel

BERT_DIR=bert-base-uncased

# DATA_DIR=./resources/data/drop/raw
DATA_DIR=/shared/nitishg/data/drop/raw


python -m bert.run_mtmsn \
  --vocab_file $BERT_DIR/vocab.txt \
  --bert_config_file $BERT_DIR/bert_config.json \
  --init_checkpoint $BERT_DIR/pytorch_model.bin \
  --do_train \
  --do_predict \
  --do_lower_case \
  --train_file $DATA_DIR/drop_dataset_train.json \
  --predict_file $DATA_DIR/drop_dataset_dev.json \
  --train_batch_size 8 \
  --predict_batch_size 8 \
  --num_train_epochs 10.0 \
  --learning_rate 3e-5 \
  --max_seq_length 512 \
  --span_extraction \
  --addition_subtraction \
  --counting \
  --negation \
  --gradient_accumulation_steps 2 \
  --seed ${SEED} \
  --output_dir ${SERIALIZATION_DIR}
