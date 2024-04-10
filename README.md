# Word2Vec
Efficient Estimation of Word Representations in Vector Space

- Summary
  - 기존에 Neural NET Language Model과 RNN Language Model 기반의 Word Representations의 시간복잡도를 분석 후 보다 효과적인 Word Representation 방법인 CBoW와 Skip-Gram을 제안
    

- Introduction
  - 그 동안의 NLP에서 사용되는 word representation method들은 단어를 atomic unit으로 다뤄서 개별적인 단어를 표현 할 뿐 그 단어들간의 유사성을 표현하지 못하였음(N-gram같은 model)
    전의 word representation method들은 large data에 대하여 상대적 적은 계산으로 처리할 수 있다는 장점이 있었으나 이러한 방식으로는 많은 task를 처리할 수 없다는 한계점이 있었음.
    
    예를 들어 특정 domain에 한정된 data의 경우 성능은 정제된 data set의 크기에 달려있음. 하지만 machine translate분야에서 여전히 많은 언어들이 충분하지 못한 scale의 말 뭉치를 가지고 있음.
    이러한 상황에서 단순한 scale up은 의미있는 변화를 가져오지 못하므로 advanced techniques를 필요로 함.

    최근 몇년간 machine learning의 발전하면서, large dataset에 대해 complex model을 훈련시키는 것이 가능해졌고 성능 또한 simple model보다 좋아짐. 대부분의 성공적인 개념들은 단어들의 분산적인
    표현(단어간의 유사성을 고려하지 않고 독립적으로 표현하는 것이 아닌 여러 차원에 걸쳐 표현 *각 차원은 특정한 의미나 특징 내포)을 이용하고 있음. 대표적으로 neural network language model은 n-gram moel보다 좋은 성능을 보임.

- 1.1 Goals of the Paper
  - 본 논문의 목표는 large dataset으로부터 high quality의 word vector를 표현하는 기술을 제안하는 것인데, 본 논문에서는 여러 차원에서 단어의 similarity를 표현하는 방법을 제안함.
    이 방법을 이용하면 vector offset연산을 통해 단어의 유사성을 표현할 수 있는데 예를 들어 king - man + woman = queen 같은 연산이 가능케 함

    단어의 선형회귀를 보존하는 새로운 model을 개발함으로 vector representation의 정확도를 최대화시켰으며 본 논문에서는 문법적인 요소와 의미적인 요소를 모두 평가할 수 있는 종합적인 test set을 설계했음. 
    또한 train에 필요한 training time까지 고려.

- 1.2 Previous Work
  - continous vector로 단어를 표현하는 연구는 긴 역사를 가지고 있음. Neural Network Language Model(NNLM)을 평가하기 위한 「A neural probabilistic language model. Journal of Machine Learning Research」
    에서 제안되었었는데 해당 model 구조에서는 linear projection을 이용하여 neural network feed forward와 non-neural hidden layer에서 vector 표현과 통계적인 language medel을 연결하여 학습에 사용함.

    또한 word vector를 학습하기 위해 처음으로 single hidden layer를 사용하는 방법이 제안된 논문들도 있으며 이런한 모델들은 full NNLM없이 word vector를 학습하는 방식을 채택함.

    -> 복잡한 model 없이도 word vector를 효과적으로 학습할 수 있으며, 이를 통해 language modeling과 관련하여 다양한 task에서 좋은 성능을 발휘함.

- 2 Model Architectures
  - 이전에 연속적인 단어 표현 추정을 위해 LSA나 LDA 같은 다양한 유형의 모델들이 제안되었었으나, 본 논문에서는 신경망에 의해 학습된 단어의 분산 표현에 초점을 맞춤.
    - LSA : 단어들 사이의 의미론적인 유사성을 분석하기 위한 통계적 기법. SVD를 이용하여 대규모의 단어-문서 행렬을 낮은 차원의 dense matrix로 축소함.
      축소된 행렬은 단어간의 의미적 유사성을 나타내는 vector로 표현됨. 이런한 방식으로 문서간의 유사성이나 의미를 파악 가능
    - LDA : 주어진 문서 집합을 topic의 혼합으로 modeling하는 확률적 생성 model. 각 문서는 다수의 topic으로 구성되어 있으며, 각 topic은 단어의 확률 분포를 가지고 있음.
      문서 내의 단어 출현 pattern을 기반으로 단어가 어떤 topic에 속할지 추론함. 이를 통해 주어진 문서의 topic 구성을 파악 가능.
      
    왜냐하면 이전 연구에서는 신경망에 학습된 분산 표현이 단어 간의 선형 관계를 보존하는 데에 LSA보다 우수하다는 것이 밝혀졌고 LDA는 large dataset에 대해서 cost가 높다는 한계가 밝혀졌기 때문임.
    유사하게, 다른 model들과 비교하여 본 논문에서는 train에 필요한 parameter의 수에 따른 계산 복잡도를 정의한다. 또한 이를 최소화하며 정확도를 높이기 위한 방법을 찾음.
    모든 model에서 training time은 다음에 비례함.


    <img width="213" alt="image" src="https://github.com/mySongminkyu/Word2Vec/assets/132251519/5dc22a45-8957-4076-882e-ec25c079348a">
    
    E = train의 epoch수, T = training set의 word수, Q = defined further for each model architecture
    
    (*typically E = 3~50, T >= one billion, 모든 model은 SGD와 backpropagation으로 훈련*)
    

    


    


