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

   - 2.1 Feedforward Neural Net Language Model(NNLM)
     - NNLM은 input,projection, hidden, output layer로 구성되어 있음.
        <img width="766" alt="image" src="https://github.com/mySongminkyu/Word2Vec/assets/132251519/d31388f4-b23f-435b-bd0e-7aee8d0edc0c">
      
        input layer에서는 one-hot encoding된 단어들이 input으로 주어진다. 총 단어의 개수는 V개, 입력 단어의 개수를 N개(총 V개의 단어내에서 중복이 있을 수 있으므로)라 하면 NxV 행렬이 만들어진다.
        이 input vector들은 각각 1xV 형태로 주어지고 VxD projection matrix에 의해 1xD(NxV proj VxD = NxD)의 vector로 사영하게 됨. -> 각각의 vector들을 모아서 하나의 거대한 word 행렬 제작

        이렇게 만들어진 NxD 행렬을 다시 hidden layer에 해당되는 DxH 행렬과 연산하여 NxH 행렬을 출력하게 되고 최종적으로 HxV output layer를 통해 NxV(각각은 1xV)로 출력되게 된다.
        여기에 softmax를 적용해주고 ground truth와의 cross entropy를 최소화하는 방식으로 진행한다.
        이러한 구조의 cost function은 다음과 같음 :

        <img width="386" alt="image" src="https://github.com/mySongminkyu/Word2Vec/assets/132251519/ebc13b5e-92f4-44fb-b7ae-1f5ed4871135">

        NNLM의 구조는 projection과 hidden layer 로 인해 연산이 복잡해지는데 N=10이라 가정하면 projection layer(P)는 500-2000이고, hidden layer(H)의 크기는 500-1000이 되게 된다.
        시간복잡도가 굉장히 높게 나오는데 이를 줄이기 위해서 hierarchical Softmax를 사용할 수 있다.

      - Hierarchical Softmax
         ![image](https://github.com/mySongminkyu/Word2Vec/assets/132251519/0000867c-9b4d-4af7-bc9e-2c1cc53e4831)

     
          다음은 어떤 단어를 중심으로 가정한 뒤 cost가 나올 확률을 binary tree를 사용하여 계산하는 것인데, 모든 node에 대하여 확률을 다 합하면 1이 나오므로 확률분포를 이루게 되며 이를 이용하면 일반적인 softmax처럼 활용할 수 있는 것이다.
          또한 binary tree를 사용하였으므로 총 단어의 개수가 V라 했을때 연산량을 $log_2V$ 로 줄일 수 있다.
          여기서 어떤 중심을 기준으로 cost를 출력하는 output matrix를 train하기 위해서는 6번,4번,3번 node의 weight만 update하면 되는데, 이처럼 연산에 관련된 node만 train하는 것은 negative sampling 방법과 유사한 특징이다.

          하지만 이러한 해결책을 써도 NxDxH 의 복잡도는 해결되지 않음 

    - 2.2 Recurrent Neural Net Language Model(RNNLM)
      - RNNLM은 RNN을 이용하여 얕은 신경망을 통해 복잡한 pattern을 표현함으로서 feedforward NNLM의 한계를 극복하고자 제안되었다.
        RNN을 이용한 model은 Projection layer를 가지지 않고 오직 input, hidden, output layer만을 가진다.

        <img width="661" alt="image" src="https://github.com/mySongminkyu/Word2Vec/assets/132251519/af2ed5d4-c292-47c2-ac8e-fca956032a6f">

        그 이유는 projection layer와 관계 없이 전의 단어들로 만들어진 context vector와 input vector만을 고려하여 output을 내보내는 model이기 때문이다.
        결과적으로 각각의 단어들에 대해 적합한 weight만을 고려하고 context는 W_H 와의 연산을 통해 만들어지기 때문에 연속적인 단어의 input을 모아주는 것을 context vector가 수행하기 때문이다.

        이처럼 RNN model의 특이점은 시간의 흐름에 따라 connect를 위해 자신의 hidden layer에 연결을 반복하는 구조이다. 따라서 계산 복잡도는 다음과 같다.

        <img width="235" alt="image" src="https://github.com/mySongminkyu/Word2Vec/assets/132251519/035ed14e-b143-4276-bfd9-11fed45929a0">

        이것 또한 HxV 부분을 Hx$log_2V$로 교체 가능

    - 2.3 Parllel Training of Neural Networks
      - large dataset에 대해 model을 훈련시키기 위해서 분산 연산 framework인 DistBlief를 사용한다. 이 framework는 같은 model을 복제시키고 각각의 복제본은 중앙 server를 통하여 기울기 update가 동기화된다.
        따라서 이는 mini-batch를 사용하고 비동기적인 방식으로 기울기를 update하며 Adagrad 최적화 알고리즘을 사용하여 network를 학습함.
        (이 framework에서는 일반적으로 100개 이상의 복제본을 이용하게 되고, multi-core 연산을 수행하게 됨)

        <img width="790" alt="image" src="https://github.com/mySongminkyu/Word2Vec/assets/132251519/0a62f29f-659f-48c8-9ff3-8e5124f3d170">

- 3 New Log-linear Models
  - 연산량을 최소화하며 단어의 분산표현을 학습할 수 있는 두가지 새로운 model을 제안, 연산의 복잡도는 non-linear(단어 간의 의미론적 유사성은 선형적인 관계로 표현 힘듦 ex_king, queen)한 hidden layer때문이었는데,
    이 때문에 단어 표현의 정확도를 줄이더라도 더욱 효율적인 훈련이 가능한 model에 대해 연구하는 것이 필요하다는 것을 밝힘. 본 논문에서 제안하는 새로운 model은 두가지 단계로 이루어져 있는데,
    먼저 단순한 model을 사용하여 연속적인 word vector를 학습한 뒤 이를 사용하여 N-gram NNLM을 학습함. 
    N-gram NNLM은 문맥을 고려하여 다음 단어를 예측하는데 사용되고 이 model은 연속적인 word vector를 input으로 받아 문장의 의미를 이해하고 다음 단어를 예측하는데에 사용됨.

- 3.1 Continuous Bag-of-Words Models(CBoW)
  - 첫 번째로 제안된 CBoW model은 hidden layer가 제거되고 모든 단어가 projection layer를 공유하고 있는 형태의 feedforward NNLM과 유사하다. 모든 단어들은 같은 position으로 projection 된다.
    우리는 이러한 구조를 Bag of Word(BoW)model이라고 하는데 이전에 porojection된 단어들은 영향을 미치지 못한다. 본 논문에서는 log-linear classifier 4개의 과거 단어와 4개의 미래 단어를 input으로
    사용하여 중심 단어가 나타나도록 훈련하여 성능을 얻고자 함. -> 주변단어로 중심단어 예측

    훈련 복잡도는 다음과 같음.

    <img width="451" alt="image" src="https://github.com/mySongminkyu/Word2Vec/assets/132251519/f0aeb383-7144-4222-9fe7-d798cdcc3193">

    NNLM 내부에 있는 projection layer의 weight matrix는 모든 단어들이 공유한다는 특성을 가지고 있고 CBoW도 이러한 특성을 갖고 진행한다.

    <img width="397" alt="image" src="https://github.com/mySongminkyu/Word2Vec/assets/132251519/8975e26b-a807-46da-b80f-45332e9501a1">

- 3.2 Continuous Skip-gram Model
  - 두 번째로 제안된 model구조는 CBoW와 유사한데, input과 output이 바뀌었다고 생각해도 좋다 -> 중심단어로 주변단어 예측
    단어 간의 간격이 멀수록 보통 단어간의 연관성이 낮아지므로 train data로 적게 부여하여 weight가 낮아지도록 설정했다.(Stochastic sampling method)
    예측 범위 증가는 word vector의 결과 quality를 향상하지만 복잡도 또한 상승하기에 적절한 범위 설정이 필요하다.

    훈련 복잡도는 다음과 같음.

    <img width="243" alt="image" src="https://github.com/mySongminkyu/Word2Vec/assets/132251519/d4fc479f-0f97-4f79-89c6-07e103ce96b8">

    C는 예측할 단어의 max distance이며 R(1-C)만큼의 개수의 단어들을 랜덤하게 선택하고 이들을 쌍으로 만들어 단어들을 학습시킨다.

    <img width="344" alt="image" src="https://github.com/mySongminkyu/Word2Vec/assets/132251519/158fed10-f375-48e5-96fb-418e3ccb8942">

- 4 Results
  - 과거의 연구는 직관적으로 단어간의 유사도가 높은 것들을 table로 제시하였는데 단어의 유사도라는 것은 vector에 있어서 단순하게 만들 수 있는 것이 아님.
    예를 들어 프랑스와 이탈리아가 국가명이라는 연관성을 만드는 것은 어렵지 않지만 확장해서 다른 나라들까지의 유사성을 만드는 것은 쉽지 않다.

    그러나 본 논문에서 제안한 model 구조를 통해 나타난 word vector들은 선형대수의 연산으로서 우리가 찾고 싶은 vector x를 구하고 그것과 cosine distance가 멀지 않은 단어를 정답으로
    채택한다. 이러한 방식의 훈련은 정확성과 효율성이 뛰어나며 large scale data set으로의 학습은 단어 사이의 미요한 의미상의 관계 학습 또한 가능했다.

    실험은 의미와 형태적 질문을 나누어서 구성했으며 질문에 대해 정확히 동일한 단어로 답변이 나올 때만 정답으로 처리함. (유사어도 불가능 따라서 100%는 불가능)

    <img width="760" alt="image" src="https://github.com/mySongminkyu/Word2Vec/assets/132251519/511ebb3f-0138-4698-82cd-5836951e9d6c">

    <img width="765" alt="image" src="https://github.com/mySongminkyu/Word2Vec/assets/132251519/797e1cbb-4850-4b56-8780-fbc268e065cd">

    <img width="766" alt="image" src="https://github.com/mySongminkyu/Word2Vec/assets/132251519/0cc0142a-1ad8-407d-850e-3a004bf84c9b">

    CBoW의 성능은 데이터 양과 차원이 동시에 높아질수록 높아졌으며 하나만 증가할 경우에는 오히려 감소하는 경향을 보임.
    
    또한 MSsoft에서 진행했던 대회에서 단일 Skip-gram 보다 Skip-gram + RNNLMs 에서 매우 높은 성능을 보임.

- 5 Summary
  - NLP에서 중요한 텍스트를 숫자로 바꾸는 방법은 여러 방법이 있지만 one-hot encoding은 차원이 너무 커지고 단어간의 semantic,syntatic 유사도를 측정할 수 없다는 단점이 있어서 Dense Representation을 사용하였다.
    이 처럼 단어를 dense vector의 형태로 표현하는 방법을 word embedding이라고 하는데 그 중 하나인 Word2Vec을 알아보았다.

    기존에도 distributional representation을 이용하여 vector를 구하는 방법은 여럿 있었지만, Word2Vec의 장점은 효율성이라 생각한다. 기존에 있던 연산량을 늘리는 layer들을 대체 및 삭제하며 간단한 neural net으로
    large scale data에 대해서도 계산량을 낮게 유지할 수 있었던 것이 크다고 본다. 

    본 논문에서 제안한 CBoW와 Skip-gram은 기존의 다른 model들과 달리 hidden layer가 존재 하지 않고 하나의 input layer와 output layer를 가져서 복잡도를 줄일 수 있다. 또한 모든 단어들이 동일한 weight matrix를 사용하여
    input layer에서 projection layer로 projection되기 때문에 projection layer 또한 없앨 수 있는 것이다.

    따라서, CBoW와 skip-gram은 간단하고 효율적인 모델 구조를 통해 단어 임베딩을 효과적으로 학습하는 데 기여하며, 이는 NLP 분야에서 중요한 발전에 임했다고 생각함.

구현 참고 : *https://github.com/rickiepark/nlp-with-pytorch/tree/main/chapter_5*












      


      
        
      








    

    


    


