##依赖
Python 3.6
scikit-learn 0.21.3
numpy 1.17.3
pandas 0.25.1
matplotlib 3.1.1

##运行方法
取消hoip.py中的注释，并运行hoip.py。


##代码和数据文件结构
│  .gitignore
│  combination_generator.py //生成未知带隙数据集
│  EDA.py                   //探索性数据分析
│  element_feature.py       //元素特征提取
│  feature_selection.py     //特征工程
│  hoip.py                  //程序入口
│  LICENSE
│  predict.py               //模型预测
│  preprocessing.py         //数据预处理和训练集-测试集分割
│  raw_plot.py              //画图
│  test_ratio_selection.py  //训练集-测试集比例优化
│           
├─data                      //数据存储目录
│  │  all_combination.csv
│  │  A_feature.csv
│  │  B_feature.csv
│  │  data_bar_plot.png
│  │  HOIP-30_drop.csv
│  │  ratio_score.csv
│  │  ratio_score.png
│  │  unknown_comb.csv
│  │  X_feature.csv
│  │  
│  ├─eda                    //EDA结果目录
│  │      pre_processing_boxplot.png
│  │      pre_processing_density.png
│  │      pre_processing_describe.csv
│  │      raw_boxplot.png
│  │      raw_density.png
│  │      raw_describe.csv
│  │      
│  └─feature                //特征作图目录
│          T_f.png
│          
├─model                     //模型储存目录
│      dtr_14_0.8616.m
│      dtr_4_0.9105.m
│      dtr_5_0.9323.m
│      feature_score_abr.png
│      feature_score_etr.png
│      feature_score_gbr.png
│      feature_score_rfr.png
│      feature_selection_abr.csv
│      feature_selection_etr.csv
│      feature_selection_gbr.csv
│      feature_selection_pca.csv
│      feature_selection_rfr.csv
│      gbr_14_0.9198.m
│      gbr_4_0.9669.m
│      gbr_5_0.9617.m
│      gpr_14_0.9523.m
│      gpr_4_0.8681.m
│      gpr_5_0.9037.m
│      krr_14_0.9281.m
│      krr_4_0.8941.m
│      krr_5_0.8712.m
│      mlp_14_0.9666.m
│      mlp_4_0.9398.m
│      mlp_5_0.9334.m
│      svr_14_0.9532.m
│      svr_4_0.8854.m
│      svr_5_0.9035.m
│      
├─predict                     //预测结果目录
│      HOIP-30_drop.csv
│      O_f.png
│      O_f_X.png
│      pred_gbr_4.csv
│      P_A_X.png
│      T_f.png
│      T_f_X.png
│      xbr.csv
│      xi.csv
│      χ_B_X.png
│      
├─training                    //模型训练代码目录
│  │  Decision_Tree.py
│  │  gbr.py
│  │  gpr.py
│  │  krr.py
│  │  MLP.py
│  │  svr.py
