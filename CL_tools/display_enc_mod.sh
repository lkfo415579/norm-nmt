ls model_revo.iter* | xargs -I {} python ~/GOD.util/performance/competence/dis_enc_mod_print.py {} encoder_Wemb > CL_MOD.log
python ~/GOD.util/performance/competence/plt_avg_mod.py BASE_AVG_MOD < CL_MOD.log
