# Path containing trained model
SERIALIZATION_DIR=/shared/nitishg/checkpoints/drop/MTMSN/date_yd_num_hmyw_cnt_whoarg_600/S_1/MTMSNModel
PREDICTION_DIR=${SERIALIZATION_DIR}/predictions

# Data root directory
DATA_DIR_ROOT=/shared/nitishg/data/drop_iclr/iclr_subm/date_ydNEW_num_hmyw_cnt_rel_600
# Path to Json file on which evaluation will be run
PREDICT_JSON=${DATA_DIR_ROOT}/drop_dataset_mytest.json

# Where to store predictions
PREDICTIONS_JSON=mytest_preds.json
METRICS_JSON=mytest_metrics.json

mkdir ${PREDICTION_DIR}

BERT_DIR=bert-base-uncased

python -m bert.run_mtmsn \
  --vocab_file $BERT_DIR/vocab.txt \
  --bert_config_file $BERT_DIR/bert_config.json \
  --init_checkpoint $BERT_DIR/pytorch_model.bin \
  --do_predict \
  --do_lower_case \
  --predict_file ${PREDICT_JSON} \
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


QUESTYPE_DATASET_DIR=questype_datasets

for EVAL_DATASET in datecomp_full year_diff_re count how_many_yards_was who_relocate_re numcomp_full
do
	PREDICT_JSON=${DATA_DIR_ROOT}/${QUESTYPE_DATASET_DIR}/${EVAL_DATASET}/drop_dataset_mytest.json
	PREDICTIONS_JSON=${EVAL_DATASET}_mytest_preds.json
	METRICS_JSON=${EVAL_DATASET}_mytest_metrics.json
	
	BERT_DIR=bert-base-uncased
	python -m bert.run_mtmsn \
  		--vocab_file $BERT_DIR/vocab.txt \
  		--bert_config_file $BERT_DIR/bert_config.json \
  		--init_checkpoint $BERT_DIR/pytorch_model.bin \
  		--do_predict \
  		--do_lower_case \
  		--predict_file ${PREDICT_JSON} \
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
	
	echo -e "METRICS: ${METRICS_JSON}"
done

