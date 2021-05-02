This repository contains the code used in the implementation of my CSE 382M Project. However, the methods used in my implementation do not belong to me.
Specifically, I take a look at four different methods and I will cite their sources as follows. Please note, at the request of the original authors, this code has only been used for purely academic purposes. Before using any code in this repository, it's important to check the README in each folder to determine where credit should be given. IN order to keep my work separate from the original author's work, they are contained in separate folders. As such, to run this code, you will either need to combine everything into one folder or add the folders to the MATLAB path. I recommend the latter. I normally uses Live Scripts to code in, but to ensure compatibility for all users, I have transferred the relevant parts of my code to MATLAB files in the "My Implementations" folder. Everything should work, but please send me a message if something is broken.

# ML-KNN
ML-KNN is a package for learning multi-label k-nearest neighbor classifiers proposed in the following paper:
Min-Ling Zhang and Zhi-Hua Zhou. ML-kNN: A lazy learning approach to multi-label learning. Pattern Recognition, 2007, 40(7): 2038-2048.
and the code contained in this repository is the exact work of the authors as obtained from [LAMDA](http://www.lamda.nju.edu.cn/code_MLkNN.ashx)
This package can only be used academically. For other purposes and questions, please contact Prof. Zhi-Hua Zhou (zhouzh@nju.edu.cn).
If this code is useful, then please cite their paper:
```
@article{
  title={ML-kNN: A lazy learning approach to multi-label learning},
  author={Zhang, Min-Ling and Zhou, Zhi-Hua},
  journal={Pattern Recognition},
  volume={40},
  number={7},
  pages={2038-2048},
  year={2007},
  publisher={Elsevier}
}
```

# BP-MLL
BPMLL is a package for training multi-label back-propogation neural networks proposed in the Following paper. 
Min-Ling Zhang and Zhi-Hua Zhou. Multilabel neural networks with applications to functional genomics and text categorization. IEEE Transactions on Knowledge and Data Engineering, 2006, 18(10): 1338-1351.
and the code contained in this repository is the exact work of the authors as obtained from [LAMDA](http://www.lamda.nju.edu.cn/code_BPMLL.ashx)
This package can only be used academically. For other purposes and questions, please contact Prof. Zhi-Hua Zhou (zhouzh@nju.edu.cn).
If this code is useful, then please cite their paper:
```
@article{
  title={Multilabel Neural Networks with Applications to Functional Genomics and Text Categorization},
  author={Zhang, Min-Ling and Zhou, Zhi-Hua},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  volume={18},
  number={10},
  pages={1338-1351},
  year={2006},
  publisher={IEEE Computer Society}
}
```

# RBRL
The RBRL method was proposed in the  following paper:  
Guoqiang Wu, Ruobing Zheng, Yingjie Tian and Dalian Liu. Joint Ranking SVM and Binary Relevance with Robust Low-Rank Learning for Multi-Label Classification. Neural Networks, 2020, 122: 24-39.
and the code contained in this  repository is the exact work of the authors cloned from their [original repository](https://github.com/GuoqiangWoodrowWu/RBRL).
If this code is useful, then please cite their paper
```
@article{wu2020joint,
  title={Joint Ranking SVM and Binary Relevance with robust Low-rank learning for multi-label classification},
  author={Wu, Guoqiang and Zheng, Ruobing and Tian, Yingjie and Liu, Dalian},
  journal={Neural Networks},
  volume={122},
  pages={24--39},
  year={2020},
  publisher={Elsevier}
}
```
and don't hesitate to contact Guoqiang Wu (guoqiangwu90@gmail.com) wtih questions

# BRWDKNN
The BRWDKNN method was proposed in the following paper:

The method of weights calculation is based on the pseudo-code presented in that paper, while the implementation is my own understanding. Currently, this implementation does not work and should be used at your own risk.  If the code looks interesting and you would like to attempt your own implementation of the pseudo-code please refere to the following details:
```
@article{wu2020joint,
  title={Joint Ranking SVM and Binary Relevance with robust Low-rank learning for multi-label classification},
  author={Wu, Guoqiang and Zheng, Ruobing and Tian, Yingjie and Liu, Dalian},
  journal={Neural Networks},
  volume={122},
  pages={24--39},
  year={2020},
  publisher={Elsevier}
}
```