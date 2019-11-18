SEED=1000
  
SERIALIZATION_DIR=/shared/nitishg/checkpoints/drop/MTMSN/date_yd_num_hmyw_cnt_whoarg_600/S_${SEED}/MTMSNModel
PREDICTION_DIR=${SERIALIZATION_DIR}/predictions

BERT_DIR=bert-base-uncased

DATA_DIR=/shared/nitishg/data/drop_iclr/date_num/date_yd_num_hmyw_cnt_whoarg_600

PREDICTIONS_JSON=mydev_preds.json
METRICS_JSON=mydev_metrics.json

python -m bert.run_mtmsn \
  --vocab_file $BERT_DIR/vocab.txt \
  --bert_config_file $BERT_DIR/bert_config.json \
  --init_checkpoint $BERT_DIR/pytorch_model.bin \
  --do_train \
  --do_predict \
  --do_lower_case \
  --train_file $DATA_DIR/drop_dataset_train.json \
  --predict_file $DATA_DIR/drop_dataset_mydev.json \
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
  --output_dir ${SERIALIZATION_DIR} \
  --prediction_dir ${PREDICTION_DIR} \
  --predictions_json ${PREDICTIONS_JSON} \
  --metrics_json ${METRICS_JSON}

