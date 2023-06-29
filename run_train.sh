#!/bin/bash


#SemEval
python re_agcn_main.py --do_train --task_name semeval --data_dir ./data/semeval/ --model_path ./bert_model_path --model_name RE_AGCN.SEMEVAL.BERT.L.doublelayer.1e-5 --do_lower_case --dep_type full_graph


