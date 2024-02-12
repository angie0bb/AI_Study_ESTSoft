# 언어 (3) - Prompting이란?

태그: HuggingFace, Prompting, 언어모델
No.: 19

### Prompt Engineering (Prompting)

- Prompt Engineering(Prompting)이란?
    - 적절한 추가정보를 제공하여 언어모델의 창의력을 사용자가 수용할 수 있는 범위로 제한하는 방법
        - 이걸 넘어서면 hallucination이라고 부른다.
- Ref: [https://ai.google.dev/docs/prompt_intro](https://ai.google.dev/docs/prompt_intro)
- 왓츠 인 마이 프롬프팅? 👜
    - Input
    - Context
    - Example

### Input

- Question Input: 질문 형태
    - What's a good name for a flower shop that specializes in selling bouquets of dried flowers?
- Task Input: 해야 할 일
    - Give me a list of things that I should bring with me to a camping trip.
- Entity Input: 정보 단위 (집합 원소 개념)
    - Classify the following items as [large, small]
    Elephant
    Mouse
    Snail
- Completion Input: 문장을 끝내지 않고 주는 것 (이어서 작성하라고)
    - 머리부터 발끝까지
    - 동해물과 백두산이

### Context

- 모델이 어떻게 행동해야 하는지 구체적으로 지시하거나
- 모델이 대답을 만들기 위한 정보나 레퍼런스를 제공하는 것
- boundary를 제한해준다.
    - 바로 green marbles이 몇 개 있는지 물어보면 추가 정보가 없기 때문에 제대로 대답을 하기 어려울 것. 따라서 그 위에 정보(context)를 제공해줄 수 있다.
        
        ![https://i.imgur.com/bkU9bmO.png](https://i.imgur.com/bkU9bmO.png)
        

### Example

- Example 제공
    
    ![https://i.imgur.com/qrr0VSM.png](https://i.imgur.com/qrr0VSM.png)
    

### Prompt design 전략

> ⭐언어 모델은 다음에 올 token을 예측하는 것
> 
- Give clear instructions
    - 명확하고 간결하게
- Include examples
    - 피해야할 패턴보다는 따라야 할 패턴을 알려주는게 좋다
    - 너무 적은 예시를 주면 overfit될 수 있음
        - 프롬프트에 예시를 몇 개 주고 학습(few shots text generation)을 시키는 것보다는 사람에 의한 튜닝이 학습이라고 생각
- Let the model complete partial input
    
    ![https://i.imgur.com/9Wz8bwF.png](https://i.imgur.com/9Wz8bwF.png)
    
    - 실습 때 만들어 둔 gemini에 돌려보니 `{cheeseburger: 2, drink: 1}`이 나왔다! cheeseburger 대신 hamburger가 나오게 하려면?
        - hamburger를 넣은 예시를 더 추가하면 된다
- Prompt the model to format its response
    
    ![https://i.imgur.com/kpWjIeU.png](https://i.imgur.com/kpWjIeU.png)
    
- Add contextual information
    - 기초적인 RAG라고 할 수 있음
        
        ```
        Answer the question using the text below. Respond with only the text provided. Question: What should I do to fix my disconnected wifi? The light on my Google Wifi router is yellow and blinking slowly.
        
        Text: Color: Slowly pulsing yellow What it means: There is a network error. What to do: Check that the Ethernet cable is connected to both your router and your modem and both devices are turned on. You might need to unplug and plug in each device again.
        
        Color: Fast blinking yellow What it means: You are holding down the reset button and are factory resetting this device. What to do: If you keep holding down the reset button, after about 12 seconds, the light will turn solid yellow. Once it is solid yellow, let go of the factory reset button.
        
        Color: Solid yellow What it means: Router is factory resetting. What to do: This can take up to 10 minutes. When it's done, the device will reset itself and start pulsing white, letting you know it's ready for setup.
        
        Color: Solid red What it means: Something is wrong. What to do: Critical failure. Factory reset the router. If the light stays red, contact Wifi customer support.
        
        ```
        
    - 이때에도 명시적으로 어떻게 정보를 써야 하는지 이야기해주면 좋다.
- Add prefixes
    - Input preflix
        - 예) `English:` , `French:`
    - Output prefix
        - 예) `JSON :`  to signal the model that the output sholud be in a JSON format.
    - Example prefix
        
        ```
        Classify the text as one of the following categories. - large - small
        Text: Rhino
        The answer is: large
        Text: Mouse
        The answer is: small
        Text: Snail
        The answer is: small
        Text: Elephant
        The answer is:
        
        ```
        

### Prompt iteration strategies

- 다른 표현을 사용해보기 (phrasing)
- 비슷한 다른 태스크로 바꿔줘보기
- 프롬프트에 넣은 내용 순서 바꿔보기
    - 질문 먼저 혹은 내용 먼저 제공해보기

### Fallback responses

- 프롬프트나 대답이 safety filter를 건드릴 때 나오는 대답
- 이때에는 temperature를 올려보면 도움이 될 수 있음

### Things to avoid

- factual information
    - 항상 가장 높은 확률의 next token이 오지 않을 수 있음. 재확인이 필요하다.
- math and logic problems
    - function calling을 사용해서 만들어줘야 제대로 대답할 수 있음

### References

- All the official gemini documents
    - [https://ai.google.dev/docs](https://ai.google.dev/docs)
    - [https://ai.google.dev/api](https://ai.google.dev/api)
- Streamlit
    - [https://docs.streamlit.io/](https://docs.streamlit.io/)
    - [https://streamlit.io/generative-ai](https://streamlit.io/generative-ai)
- Gemini Pro + LangChain
    - RAG로 넣을 document가 많지 않으면 pandas 나 벡터로 가지고 있어도 되지만, 양이 많아지면 DB도 필요할 거고 그 안에서 벡터 유사도를 빠르게 계산하기 위한 패키지가 필요할 것. -> 모아놓은 패키지가 LangChain
        - LlamaIndex 도 있음
    - [https://python.langchain.com/docs/templates/rag-gemini-multi-modal](https://python.langchain.com/docs/templates/rag-gemini-multi-modal)
    - [https://youtu.be/G3-YOEVg-xc?feature=shared](https://youtu.be/G3-YOEVg-xc?feature=shared) (15:08)

### Summary

![https://i.imgur.com/Et0MjKh.png](https://i.imgur.com/Et0MjKh.png)

## Hugging Face

### phi-2

- [colab 실습](https://colab.research.google.com/drive/17Dvs97jdX_TznzdcqkCcVDxSGm2M8Xbv#scrollTo=TTHrwa3g5FGx)
- hugging face model card ([link](https://huggingface.co/microsoft/phi-2))
- fine-tuning할때는? add-on을 붙혀서 사용할 수 있다.
    - LoRA ([github](https://github.com/microsoft/LoRA)): 비교적 적은 비용으로 할 수 있음 (A100 한 대 기준 7-8시)
        - 비전에서도 쓸 수 있음. diffusion 모델에다가 새로운 컨셉을 학습 시킬 때 LoRa를 사용하기도 한다.

### SBERT

- [https://github.com/snunlp/KR-SBERT](https://github.com/snunlp/KR-SBERT)

### BERT

- [https://neptune.ai/blog/how-to-code-bert-using-pytorch-tutorial](https://neptune.ai/blog/how-to-code-bert-using-pytorch-tutorial)