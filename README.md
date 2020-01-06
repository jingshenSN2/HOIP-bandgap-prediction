HOIP_bandgap_prediction
===
依赖环境
---
Python 3.6<br>
scikit-learn 0.21.3<br>
numpy 1.17.3<br>
pandas 0.25.1<br>
matplotlib 3.1.1<br>

运行方法
---
取消hoip.py中的注释，并运行hoip.py。<br>


代码和数据文件结构
---
│  .gitignore<br>
│  combination_generator.py //生成未知带隙数据集<br>
│  EDA.py                   //探索性数据分析<br>
│  element_feature.py       //元素特征提取<br>
│  feature_selection.py     //特征工程<br>
│  hoip.py                  //程序入口<br>
│  LICENSE<br>
│  predict.py               //模型预测<br>
│  preprocessing.py         //数据预处理和训练集-测试集分割<br>
│  raw_plot.py              //画图<br>
│  test_ratio_selection.py  //训练集-测试集比例优化<br>
│           <br>
├─data                      //数据存储目录<br>
│  │  all_combination.csv<br>
│  │  A_feature.csv<br>
│  │  B_feature.csv<br>
│  │  data_bar_plot.png<br>
│  │  HOIP-30_drop.csv<br>
│  │  ratio_score.csv<br>
│  │  ratio_score.png<br>
│  │  unknown_comb.csv<br>
│  │  X_feature.csv<br>
│  │  <br>
│  ├─eda                    //EDA结果目录<br>
│  │      pre_processing_boxplot.png<br>
│  │      pre_processing_density.png<br>
│  │      pre_processing_describe.csv<br>
│  │      raw_boxplot.png<br>
│  │      raw_density.png<br>
│  │      raw_describe.csv<br>
│  │      <br>
│  └─feature                //特征作图目录<br>
│          T_f.png<br>
│          <br>
├─model                     //模型储存目录<br>
│      dtr_14_0.8616.m<br>
│      dtr_4_0.9105.m<br>
│      dtr_5_0.9323.m<br>
│      feature_score_abr.png<br>
│      feature_score_etr.png<br>
│      feature_score_gbr.png<br>
│      feature_score_rfr.png<br>
│      feature_selection_abr.csv<br>
│      feature_selection_etr.csv<br>
│      feature_selection_gbr.csv<br>
│      feature_selection_pca.csv<br>
│      feature_selection_rfr.csv<br>
│      gbr_14_0.9198.m<br>
│      gbr_4_0.9669.m<br>
│      gbr_5_0.9617.m<br>
│      gpr_14_0.9523.m<br>
│      gpr_4_0.8681.m<br>
│      gpr_5_0.9037.m<br>
│      krr_14_0.9281.m<br>
│      krr_4_0.8941.m<br>
│      krr_5_0.8712.m<br>
│      mlp_14_0.9666.m<br>
│      mlp_4_0.9398.m<br>
│      mlp_5_0.9334.m<br>
│      svr_14_0.9532.m<br>
│      svr_4_0.8854.m<br>
│      svr_5_0.9035.m<br>
│      <br>
├─predict                     //预测结果目录<br>
│      HOIP-30_drop.csv<br>
│      O_f.png<br>
│      O_f_X.png<br>
│      pred_gbr_4.csv<br>
│      P_A_X.png<br>
│      T_f.png<br>
│      T_f_X.png<br>
│      xbr.csv<br>
│      xi.csv<br>
│      χ_B_X.png<br>
│      <br>
├─training                    //模型训练代码目录<br>
│  │  Decision_Tree.py<br>
│  │  gbr.py<br>
│  │  gpr.py<br>
│  │  krr.py<br>
│  │  MLP.py<br>
│  │  svr.py<br>
