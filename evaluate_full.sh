# Path containing trained model

MTMSN_BASE=/shared/nitishg/checkpoints/drop/MTMSN/full_data/S_10/MTMSNModel

SERIALIZATION_DIR=${MTMSN_BASE}
PREDICTION_DIR=${SERIALIZATION_DIR}/predictions

# Path to Json file on which evaluation will be run
DEV_DATA_JSON=/shared/nitishg/data/drop/raw/drop_dataset_dev.json

DEV_DATA_JSON=/shared/nitishg/minimal-pairs/final_annotations/minimal_pairs_test.json

# Where to store predictions
PREDICTIONS_JSON=minimal_pairs_test_preds.json
METRICS_JSON=minimal_pairs_test_metrics.json

mkdir ${PREDICTION_DIR}

BERT_DIR=bert-base-uncased

python -m bert.run_mtmsn \
  --vocab_file $BERT_DIR/vocab.txt \
  --bert_config_file $BERT_DIR/bert_config.json \
  --init_checkpoint $BERT_DIR/pytorch_model.bin \
  --do_predict \
  --do_lower_case \
  --predict_file ${DEV_DATA_JSON} \
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
  --output_dir ${SERIALIZATION_DIR} \
  --prediction_dir ${PREDICTION_DIR} \
  --predictions_json ${PREDICTIONS_JSON} \
  --metrics_json ${METRICS_JSON}
