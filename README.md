# AI Repository
MUNA 팀의 CreateTrend 프로젝트에서 영상의 트렌드 정보를 분석하고 조회수 예측 딥러닝 모델을 구현한 레포입니다.  

## PipeLine
### 영상 컨텐츠 트렌드 분석 기능
![image](https://user-images.githubusercontent.com/50457791/99808482-48ae6c00-2b84-11eb-9e87-21709b262c46.png)  
1. 영상 조회수, 채널 구독자수, 채널 전체 조회수를 통계분석해 인기 지수를 계산합니다.
2. 인기지수를 컨텐츠와 썸네일 장르별로 할당합니다.
3. 컨텐츠 목록을 HDBScan을 이용한 클러스터링으로 컨텐츠 카테고리를 생성하고 카테고리별 인기 지수를 계산합니다.
4. Facebook Prophet 라이브러리를 활용해 인기지수를 예측하고 컨텐츠, 카테고리, 썸네일 정보를 종합해 트렌드 지수를 계산합니다.

### 조회수 예측 기능
![image](https://user-images.githubusercontent.com/50457791/99808488-4c41f300-2b84-11eb-82ed-5a9db8d1b2f5.png)  
1. 영상 썸네일을 BiT-M R101x1 모델을 이용해 2048 차원의 Feature Vector를 추출합니다.
2. 영상 제목을 YouTube 데이터로 학습시킨 KoBert 모델을 활용해 768 차원의 Feature Vector를 추출합니다.
3. 채널 구독자수와 영상 업로드 날짜를 Normalize해 4개의 벡터를 통합합니다.
4. 예측 모델을 통해 조회수를 예측합니다.

## Built With
현 프로젝트는 다음의 주요 서비스를 통해 개발되었습니다.
* [Facebook Prophet](https://facebook.github.io/prophet/)
* [BiT-M R101x1](https://tfhub.dev/google/bit/m-r101x1/1)
* [KoBERT](https://github.com/SKTBrain/KoBERT)
* [Tensorflow](https://www.tensorflow.org/?hl=ko)

## 기능 상세 설명
### 영상 컨텐츠 트렌드 분석 기능
![image](https://user-images.githubusercontent.com/50457791/99808497-4ea44d00-2b84-11eb-9244-8b464f0c0b0e.png)  
단순 조회수가 높은 영상의 경우는 구독자수가 많은 영상으로 편중됩니다.  
채널의 영상을 자주 올리는 크리에이터와 간간이 올리는 크리에이터의 조회수는 구독자수 대비 조회수 비율이 차이납니다.  
위 경향을 반영하기 위해 각 영상의 조회수를 구독자수로 나눈 값을 계산하고 채널 전체 조회수 증가수 대비 영상 조회수를 계산해  
해당 영상이 채널에 갖는 비중을 곱한 인기 지수를 계산합니다.  
해당 인기지수를 분석 기간동안 등장한 컨텐츠별로 부여해 모든 영상의 인기 지수가 반영된 컨텐츠별 인기 지수를 계산합니다.  
`위 프로세스는 모두 DB의 트리거 동작으로 구현되어 본 레포에 소스가 존재하지 않습니다.`  

*Facebook Prophet 기능은 현재 구현 테스트 단계이며 데이터 부족으로 향후 적용 예정입니다.*

### 조회수 예측 기능
![image](https://user-images.githubusercontent.com/50457791/99808505-5106a700-2b84-11eb-8af1-941a85adb34e.png)  
유튜브의 추천 시스템 딥러닝 모델(Reference 참조)을 참고해 조회수 예측 모델을 구현했습니다.  
시청자가 영상 목록에서 해당 영상을 볼지 말지 결정은 영상 썸네일, 제목, 채널 정보로 결정하며 해당 영상이 추천시스템에 등재될지는 조회수와 영상 날짜가 반영됩니다.  
해당 요소들의 영향을 반영하기 위해 썸네일은 CNN, 제목은 NLP를 통해서 피쳐 벡터를 추출하고 채널 구독자수와 업로드 날짜를 입력으로 조회수를 예측합니다.  

모델의 Loss 값과 MAE 값 모두 0.001로 수렴하였으며 이는 Log Scaling이 적용되어있기에  
실제 수치는 약 5%의 조회수 오차를 보인다고 평가할 수 있습니다.  

## Reference 
#### YouTube Recommendation System 
![image](https://user-images.githubusercontent.com/50457791/99808510-53690100-2b84-11eb-89c7-3196a67a8d93.png)  
[Deep Neural Networks for YouTube Recommendations](https://dl.acm.org/doi/abs/10.1145/2959100.2959190)  
본 네트워크를 참조해 영상 조회수 예측 모델을 구현하였습니다.

## Authors
- **박지홍(qkrwlghddlek@naver.com)**
