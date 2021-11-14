# Transformer

- Attention 만을 이용해서 SOTA를 이끌어낸 연구
    - RNN의 long-term dependency problem 를 attention mechanism만을 사용해 input과 output의 dependency를 포착하여 해결
- Seq2Seq와 유사한 구조
- RNN의 역전파 과정이 없으므로 병렬 계산이 가능함
- Positional encoding을 사용함(입력 단어의 위치를 표현, context 이해 향상)

# Model Architecture
<img src="/images/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2021-11-14_15.19.42.png" width="50%" height="50%">

- Encoder input : $X = [X_0, X_1, ..., X_n]$
- Decoder
    - input(shifted right of Encoder output)
    - output : 소프트맥스를 통한 Output Probabilities
- Positional Encoding : 상대적인 위치에 따라서 고유의 벡터를 생성하여 input에 더해줌
    
    $PE_(pos, 2i) = sin(pos/10000^{2i/d_{model}})$
    
    $PE_(pos, 2i+1) = cos(pos/10000^{2i/d_{model}})$
    

# Scaled Dot-Product Attention
<img src="/images/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2021-11-14_15.33.50.png" width="50%" height="50%">

- Comparison
: $C = softmax(\frac{K^TQ}{\sqrt{d_k}})$
- Multi-Head Attention
    - Q, K, V의 차원을 감소시킴
    - h개의 layer로 나누어 병렬적으로 연산 출력시에 linear 연산을 통해서 출력 단어 종류의 수에 맞춤
    : $Linear_i(V,K,Q) = (V,K,Q)W_{(V,K,Q), i} \in \R^{d_v \times d_{model}}$
- Mask(opt.)
: 자기 자신을 포함한 미래의 값과는 attention을 구하지 않기 때문에 masking을 사용하여 V의 범위를 제한시키는 필터
