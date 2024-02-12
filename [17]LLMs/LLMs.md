# 언어 (1) - LLMs이란?

태그: Gemini, LLMs, 언어모델
No.: 17

## LLM and Prompting

- ⭐실습 [colab](https://colab.research.google.com/drive/1mrn8UwCRRdtTiOqnGj6Xf4Upo-N-yKoz#scrollTo=sVrm7VyTECQA)
- 다룰 것
    - Python SDK for Google Gemini Pro
    - Prompting
    - Retriever Augmented Generation
    - Streamlit
    - Hugging Face

### AI Re-visit

- AI algorithms are just Approximation Functions
    - 모든 것은 단지 함수를 근사하는 것 뿐이다!
- 인간의 판단력을 모사하는 인공지능
    - classification, object detection, segmentation
    - 자율주행, 안면인식, ...
    - 정확한 하나의 정답을 내야 함
- 인간의 창의력을 모사하는 인공지능
    - LLM: ChatGPT(OpenAI), Gemini(Google), LLaMA2(Meta), ...
    - Diffusion: DALL-E(OpenAI), Stable Diffusion(Stability AI), ...
    - 인간이 수용할 수 있는 정답, 목표를 이루기 위한 어떤 정답
        - 그러다 보니 평가를 사람이 한다. (Metrics 정하기가 어려움)

### Large Language Models(LLMs)

- 필요한 것
    1. a method for evaluation
        - human feedback: 비쌈
        - GPT4 feedback: 비용절감을 위해 어쩔수없이 사용, 하지만 GPT4보다 뛰어난 모델 만들긴 어려움.
            - 서로 다른 데이터셋에 대해 여러가지 모델을 학습하고, 이를 ensemble하는 방법으로 극복하려고 함
    2. candidate models
        - some variations of Transformer: Attention is all you need (2017)
    3. a way to find the parameters:
        - predict the next token
            - train dataset: Wikipedia, Github, News papers, ...
            - real valued differentiable cost function: cross-entropy
            - gradient descent algorithm
            - RLHF (Reinforcement learning from human feedback)
- 멀티모달?
    
    ![https://i.imgur.com/BgrdePI.png](https://i.imgur.com/BgrdePI.png)
    
- 어려운 점
    - 양질의 train dataset 얻기가 어렵다.
        - 신문기사 -> 양질의 dataset이지만 요즘은 인공지능에 활용하는 걸 금지하는 경우가 많음.
        - 네이버 클로바: 블로그, 영화평론, 댓글
        - 한글 데이터 자체가 적다.
    - 우리나라는 정보 공유에 인색한 면이 조금 있음.
- **언어 모델의 원리: 다음에 올 단어(Token)을 예측**
    - 단점이 정말 단점일까? 오히려 창의력을 보여줄 수 있음.
        
        ![https://i.imgur.com/AEzdBMD.png](https://i.imgur.com/AEzdBMD.png)
        
    - 단점 -> 적절한 추가 정보를 제공하면, 하나의 정답이 생김

### Prompt Engineering (Prompting)

- ==Prompt Engineering(Prompting)이란?==
    - 적절한 추가정보를 제공하여 언어모델의 창의력을 사용자가 수용할 수 있는 범위로 제한하는 방법
        - 이걸 넘어서면 hallucination이라고 부른다.
- Google AI Studio
    - [https://ai.google.dev](https://ai.google.dev/)
    - 개인용 구글 id로 접속
    - 가끔 불안정할 때를 대비해 서비스하는 경우에는 try, except, retry를 위한 decorator를 써주면 좋다.
- Basic Usage
    - Install SDK
        
        ```python
        pip install -q -U google-generativeai
        
        ```
        
    - Import package
        
        ```python
        import google.generativeai as genai
        
        ```
        
    - API Key
        
        ```python
        genai.configure(api_key=GOOGLE_API_KEY)
        
        ```
        
- Model Parameters
    
    ```python
    cfg = genai.GenerationConfigt(
    	candidate_count = 1 # 답변의 개수 (고정되어있음
    	stop_sequences = None, # 해당 문자열을 만나면 무조건 생성 중단, 최대 5개 리스트 형식 문자열
    	max_output_tokens = None, # 토큰 개수를 몇 개나 생성할건지, 최대 2048개 (pro 기준), 말이 너무 길어지지 않게 조정하는 역할을 함
    	temperature = 0.9, # temperature scaling: 성능향상을 위해 classification에서 softmax 직전에 어떤 값을 곱해주거나 나눠주는 방법
    	top_k = 40
    	top_p = 0.95
    	# 보통 temp만 건들이고 top_k, top_p는 잘 건드리지는 않는다
    
    ```
    
    - Max output tokens
        - Maximum number of tokens that can be generated in the response.
        - A token is approximately four characters. 100 tokens은 대략 60-80개의 단어로 이루어져있다.
    - ⭐ Temperature
        - temperature scaling: 성능향상을 위해 classification에서 softmax 직전에 어떤 값을 곱해주거나 나눠주는 방법
        - temperature: temperature controls the degree of randomness in token selection
            - 언어모델이 출력은 다음에 올 token의 확률값들
                - I 다음에 be 동사가 올 확률: 0.3
                - do 동사가 올 확률: 0.2
                - work 동사가 올 확률: 0.1 등
                - 최종적으로는 이 확률에 맞추어서 sampling을 하는 것
            - 이러한 확률값을 만들기 위해 모델의 제일 밑단에 있는 값들을 softmax함수에 집어넣어줄 것.
                - softmax의 역할: 값들을 전부 1보다 작게, 다 합했을 때 1로 만들어준다.
            - 나올 수 있는 모든 TOKEN값에 대해 확률을 계산하는 시도를 할 것 -> 매번 너무 많은 계산이 필요해진다.😭
                - `Top-K (most probable)`: default = 40, softmax에 집어 넣는 값을 `logits`이라고 한다. logits은 학습에 사용한 token의 개수만큼 있엄. 이 중에 logits이 큰 것 K개를 뽑는다. -> softmax에서 순서가 바뀌지 않기 때문에, softmax를 거쳐 나온 확률이 제일 높은 k개와 동일할 것
                - 이 top-K logits을 temperature로 나누어준 다음 softmax에 넣는다.
                    - 작은 값으로 나누기 때문에 결과는 매우 커질 것.
                    - 큰 값으로 나누면 결과는 매우 작아지고, 이걸 softmax에 넣으면 판판해질 것
                    - **즉, 창의력(답변이 다양하면 좋겠다)을 키우고 싶으면 temp(나누는 값)을 크게, 그 반대의 경우(그럴듯한 답만 뽑아내고 싶다)에는 temp를 작게 주면 된다.**
                - `Top-P`: default = 0.95, tokens are selected from the most (top-K) to least probable until the sum of their probabilities equals the top-P value. Spcify a lower value for less random responses and a higher value for more random responses.
                    - tokens A = 0.3, B = 0.2, C = 0.1, D= 0.1 이고 Top-p = 0.5면, A, B까지만 선택하고 C, D는 선택 안함. 크게 줄수록 창의력 or randomness이 커짐.
            - 모델은 다음에 올 값을 동사 중에서 뽑겠다!라고 사고하는 것이 아닌, 노래를 외우는 것처럼 다음에 올 값을 자연스럽게! 외운 것들 중에서 내보낸다. (에: 동해물과 -> 백두산이, 독도는 -> 우리땅, 독도는 -> 아름답다 (이것도 문제는 없음, 독도는 우리땅만 나오면 안 되고 이런 문장도 나올 확률이 존재해야 함))
- Generate Text from text imputs
    
    ```python
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content("동해물과 ", generation_config=cfg)
    ```
    
- Generate Text from image and text inputs
    - PIL, jpg 등 아무거나 넣어도 잘 처리해준다.
        
        ```python
        model = genai.GenerativeModel('gemini-pro-vision')
        response = model.generate_content(
        	  [
        		  "Transform the following recipt in a markdown table in English."
        		  img  # just put PIL image, bytes code with mime type, etc.
        	  ],
        	  generation_config=cfg
        )
        
        ```
        
- Response format
    
    ```python
    response.candidates
    
    ```
    
    ```python
    [content{
    	parts {
    		text: MODEL RESPONSE TEXT HERE
    	}
    	role: "model"  # model or user
    	}
    finish_reason: STOP or SAFITY # STOP이 정상, 모델이 문장을 끝내서 STOP을 주거나 stop_sequences에 있는 단어를 만났을 때 STOP을 준다. NETWORK ERROR, INTERNAL ERROR 등도 넣을 수 있음. SAFITY: 혐오, 차별 발언 등을 safety_ratings로 검열
    index: 0
    saftey_ratings{ # 생성한 것에 대해서도 계속 평가를 한다
    		category: HARM_CATEOGRY_{}
    		probability: {}
    }]
    
    ```
    
- To get text
    
    ```python
    response.candidates[0].content.parts[0].text
    
    ```
    
- Short cut for getting text
    
    ```
    response.text
    # does not work if `stream=True` in `model.generate_content()`
    
    ```
    
- For stream output: 사람처럼 글자를 하나씩 찍어주는 것 , 속도 차이는 없음
    
    ```python
    for chunk in response:
    	# print(chunk.candidates[0].content.parts[0].text)
    	# shortcut works here! (after loop iteration)
    	print(chunk.text)
    
    ```
    
    - for debugging
        
        ![https://i.imgur.com/tvaDJbh.png](https://i.imgur.com/tvaDJbh.png)
        
- Generate text from image and text inputs (멀티모달)
    - (경험적 tip) temp를 좀 작게 주는 것이 좋다.
        
        ```python
        !curl -o image.jpg <https://upload.wikimedia.org/wikipedia/commons/0/0b/ReceiptSwiss.jpg> # 이미지 하나 불러오기
        
        ```
        
        ```python
        import PIL.Image
        img = PIL.Image.open('image.jpg')
        img
        ```
        
        ```python
        cfg = genai.GenerationConfig(
            candidate_count = 1,
            stop_sequences = None,
            max_output_tokens = None,
            temperature = 0.1,  # 작게
            top_k = 32,
            top_p = 0.5,
        )
        model = genai.GenerativeModel('gemini-pro-vision')
        response = model.generate_content(
            [
                "Transform the following recipt in a markdown table in English.",
                img
            ], generation_config=cfg, stream=True)
        response.resolve()
        to_markdown(response.text)
        
        ```
        

### Embedding generation

- 참고 document ([link](https://www.notion.so/angieeee/%5B%3Chttps://ai.google.dev/examples/doc_search_emb?hl=en%3E%5D(%3Chttps://colab.research.google.com/corgiredirector?site=https%3A%2F%2Fai.google.dev%2Fexamples%2Fdoc_search_emb%3Fhl%3Den%3E)))
- retrieval augmentation generation의 핵심!
    - 모든 문서를 다 인풋으로 받아줄 수 없으니, 질문과 유사한 문서를 선택해서 (cosine similarity 이용) 가져온다.
- Task types
    - retrieval_query: 질문을 벡터로
    - retrieval_document:
        - 질문에 대한 답이 들어있을 확률이 높은 document를 선택
            - Consine similarity, dot product 값 등으로 유사도 평가
    - semantic_similarity: 의미가 비슷한 것끼리 모으는 용도 (이것도 벡터로)
    - classification: 분류에 사용하기 위한 벡터
    - clustering: 비슷한 것끼리 모으기 위한 벡터
- code
    
    ```python
    import numpy as np
    import pandas as pd
    
    DOCUMENT1 = {
        "title": "Operating the Climate Control System",
        "content": "Your Googlecar has a climate control system that allows you to adjust the temperature and airflow in the car. To operate the climate control system, use the buttons and knobs located on the center console.  Temperature: The temperature knob controls the temperature inside the car. Turn the knob clockwise to increase the temperature or counterclockwise to decrease the temperature. Airflow: The airflow knob controls the amount of airflow inside the car. Turn the knob clockwise to increase the airflow or counterclockwise to decrease the airflow. Fan speed: The fan speed knob controls the speed of the fan. Turn the knob clockwise to increase the fan speed or counterclockwise to decrease the fan speed. Mode: The mode button allows you to select the desired mode. The available modes are: Auto: The car will automatically adjust the temperature and airflow to maintain a comfortable level. Cool: The car will blow cool air into the car. Heat: The car will blow warm air into the car. Defrost: The car will blow warm air onto the windshield to defrost it."}
    
    DOCUMENT2 = {
        "title": "Touchscreen",
        "content": "Your Googlecar has a large touchscreen display that provides access to a variety of features, including navigation, entertainment, and climate control. To use the touchscreen display, simply touch the desired icon.  For example, you can touch the \\"Navigation\\" icon to get directions to your destination or touch the \\"Music\\" icon to play your favorite songs."}
    
    DOCUMENT3 = {
        "title": "Shifting Gears",
        "content": "Your Googlecar has an automatic transmission. To shift gears, simply move the shift lever to the desired position.  Park: This position is used when you are parked. The wheels are locked and the car cannot move. Reverse: This position is used to back up. Neutral: This position is used when you are stopped at a light or in traffic. The car is not in gear and will not move unless you press the gas pedal. Drive: This position is used to drive forward. Low: This position is used for driving in snow or other slippery conditions."}
    
    df = pd.DataFrame([DOCUMENT1, DOCUMENT2, DOCUMENT3])
    df.columns = ['Title', 'Text']
    df
    ```
    
    ```python
    from google.api_core import retry
    
    # Get the embeddings of each text and add to an embeddings column in the dataframe
    
    @retry.Retry(timeout=300.0)  # 실패하면 5분 동안 재시도해보라는 retry decorator
    def embed_fn(title, text): # 문서의 제목과 내용을 받아서 벡터 만드는 함수
      return genai.embed_content(model='models/embedding-001', # 임베드용 모델 이름 지정
                                 content=text,
                                 task_type="retrieval_document",
                                 title=title)["embedding"]
    
    ```
    
    ```python
    df['Embeddings'] = df.apply(lambda row: embed_fn(row['Title'], row['Text']), axis=1)
    df
    
    ```
    
    ```python
    len(df['Embeddings'].iloc[0]) # 길이가 768인 벡터로 만들어줌
    
    ```
    
    ```python
    query = "How do you shift gears in the Google car?"
    request = genai.embed_content(model='models/embedding-001',
                                  content=query,
                                  task_type="retrieval_query")
    
    ```
    
    ```python
    len(request['embedding'])
    
    ```
    
    ```python
    dot_products = np.dot(np.stack(df['Embeddings']), request['embedding']) # 임베딩 컬럼에 있는 것들을 쌓아주고, request임베딩에 있는 것들로 dot product해줌
    idx = np.argmax(dot_products)
    to_markdown('## ' + df.iloc[idx]['Title'] + '\\n\\n' + df.iloc[idx]['Text'])  # 질문에 대한 가장 연관성이 높은 document인 Shifting Gears를 뽑아주었다!
    
    ```
    
    ```python
    dot_products # 뒤에 있는 문서가 값이 더 높다!
    # output: array([0.66797978, 0.66149501, 0.77680849])
    
    ```
    

### Question and Answering Application

- 위에 배운 내용을 접목시켜보자!
    
    ```python
    def find_best_passage(query, dataframe):
    
      """
      Compute the distances between the query and each document in the dataframe
      using the dot product.
      """
      query_embedding = genai.embed_content(model='models/embedding-001',
                                            content=query,
                                            task_type="retrieval_query")
      dot_products = np.dot(np.stack(dataframe['Embeddings']), query_embedding["embedding"])
      idx = np.argmax(dot_products)
      return dataframe.iloc[idx]['Text'] # Return text from index with max value
    
    def make_prompt(query, relevant_passage): # 질문(query), relevant_passages - 위의 find_best_passage에서 뽑힌, 가장 연관성이 높은 document를 같이 넣어준다.
      escaped = relevant_passage.replace("'", "").replace('"', "").replace("\\n", " ")
      prompt = textwrap.dedent("""You are a helpful and informative bot that answers questions using text from the reference passage included below. \\
      Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. \\
      However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \\
      strike a friendly and converstional tone. \\
      If the passage is irrelevant to the answer, you may ignore it.
      QUESTION: '{query}'
      PASSAGE: '{relevant_passage}'
        ANSWER:
      """).format(query=query, relevant_passage=escaped)
      return prompt
    
    ```
    
    ```python
    cfg = genai.GenerationConfig(
        candidate_count = 1,
        stop_sequences = None,
        max_output_tokens = None,
        temperature = 0.,  # 언어/생성 모델에서 시연할 때에는 temp를 0으로 매우 낮게 주자..
        top_k = 40,
        top_p = 1.,
    )
    
    ```
    
    ```python
    query = "How do you shift gears in the Google car?"
    passage = find_best_passage(query, df)
    prompt = make_prompt(query, passage)
    
    model = genai.GenerativeModel('gemini-pro')
    answer = model.generate_content(prompt, generation_config=cfg)
    to_markdown(answer.text)
    
    ```
    
    ```python
    # prompt없이 그냥 물어보면? hallucination을 보여준다..
    answer = model.generate_content(query, generation_config=cfg)
    to_markdown(answer.text)
    
    ```
    
- 이렇게 적절한 추가 정보(prompting)을 주고 답변을 이끌어내는 걸 RAG (Retriever Augmentation Generation)라고 한다.
- 원리만 일단 살펴봄
    - 실제로 큰 작업을 해야할 때에는 여러가지 패키지를 이용할 수 있다.

### Multi-turn Conversation (Chat)

- code
    
    ```python
    model = genai.GenerativeModel('gemini-pro')
    chat = model.start_chat(history=[])
    response = chat.send_message("In one sentence, explain how a computer works to a young child.")
    to_markdown(response.text)
    
    ```
    
    ```python
    chat.history # 히스토리 찍어보기
    # chat.history[-1]  # 마지막 줄만
    
    ```
    
    ```python
    response = chat.send_message("Okay, how about a more detailed explanation to a high schooler?", stream=True) # stream 그냥 걸어본 것
    for chunk in response:
      print(chunk.text)
      print("_"*80)
    
    ```
    
    ```python
    for message in chat.history:
      display(to_markdown(f'**{message.role}**: {message.parts[0].text}'))
    
    ```
    

`