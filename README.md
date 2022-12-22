### Title:
2022 **Travel Data** Analysis

### Team: (**individual team**)
| Number        | Name          | Major        |
| ------------- |:-------------:|:-------------|
| 20102056      | Hwang inseo   | IISE & computer engineering         |

### Description:
1. Task :
    1. classification task
    1. multi-modal task
2. PurPose :
    - Create an algorithm that can use images and data to automatically classify categories
4. Why :
    - I'm interested in multi-modal
    - This is likely to be good data for beginners who start multimodal.
    
### Progress
- image : 이미지 데이터에 있어서 다른 전처리 진행하지 않음
    - 사이즈와 화질이 모두 다르다는 것을 보고 필요성이 보였다.
    - 이후 성능을 올릴때 augmentation을 시도해 봐야겠다.
- text : 
    - tokenization을 통해 모델에 넣어주었다.
    - token화 과정은 Hugging Face에서 지원하는 AutoTokenizer을 사용해 진행했다. -> 명사와 감탄사 추출
    - max length, padding truncation 모두 적용하고 True로 설정했다.
        
### Model
- image : pretrained 모델인 Vit 모델 사용! (32개의 patch로 나눠져 들어가는 모델이다.)
    - 사용 이유 : efficient-Net을 사용해보려고 했었는데, Vit가 성능이 좋다고 해서 사용하게 되었다.
    - 그러나 transformer모델의 경우 데이터 셋이 작을 때의 단점이 존재한다는 것을 알게 되었다.
    ![vit image](https://user-images.githubusercontent.com/90232567/206888275-9e14532e-d5cd-41a3-8538-cddef9ec57e4.png)
- text : Roberta 모델 사용
    - 사용 이유 : bert 모델을 공부하다 roberta가 bert모델의 단점을 보완한 모델이라서 사용해 보았다.
    ![roberta model image](https://user-images.githubusercontent.com/90232567/206888343-decfa103-44e9-474d-8d1c-d05f7a164579.png)
- fusion : 여러 방법이 있다는 것을 알게 되었다.
    - concat
    - multiple
    - transformer : 현재 연구가 가장 많이 진행되고 있다고 한다.
    - 여기선 concat 후, encoder layer를 통과시켰다.
        ![fusion image](https://user-images.githubusercontent.com/90232567/206889616-89548a49-27e1-4f73-95b2-91288e8ef7aa.png)
- loss function : 가중치를 적용해서 설정 (이전 연구 참고)
    - ![Untitled (3)](https://user-images.githubusercontent.com/90232567/206888448-0ce1866f-d548-48bc-ac9b-213697d66eb5.png)
    
### Result
- train loss 0.17, train accuracy 0.96
- validation loss 0.64, validation accuracy 0.83
- ![result image](https://user-images.githubusercontent.com/90232567/206891633-cede7a79-47fc-4a4e-bb30-b21b7a48f44e.png)


### Conclusion
- 특이사항 : 데이터 셋이 크지 않아서 layer를 늘리면 성능이 떨어지는 경향이 있다.
- 한계점 :
    - 전처리에 있어서 많은 아쉬움이 남는다. 조금 더 공부를 함으로 전처리를 진행했으면 좋았을 것 같다.
    - 모델 사용 방법을 익힌 것이지 모델을 뜯어서 이해한게 아니라 모델의 깊은 이해를 못한게 아쉽다.



