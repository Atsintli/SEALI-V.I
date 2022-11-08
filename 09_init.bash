#/bin/bash
export models=(04_violin_model_4k)

puerto=8531
for model in ${models[@]}
do
  tensorflow_model_server --rest_api_port=$puerto --model_name=improv_class --model_base_path=$(pwd)/$model &
  ((puerto)) #para incrementar el numero de puerto
done

sleep 1
python3 08_extract_features_OSC_sender_violin.py &
sleep 3
python3 07_rt_features_extraction_LSTM_close_candidate_OSC.py

