# 언어 (2) - Streamlit Chatbot을 만들어보자

태그: Streamlit, 언어모델
No.: 18

Reference: [Streamlit Docs](https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps)

- watchdog이 같이 깔려져있으면, py파일을 수정하고 저장했을 때에 웹 페이지에서 streamlit을 재실행하지 않아도 새로고침만으로 바로 반영이 된다.
- api-reference: [https://docs.streamlit.io/library/api-reference/data](https://docs.streamlit.io/library/api-reference/data)
    - 프로젝트나 포트폴리오 작성할때 visualization으로 쓰면 좋다.
- 정보를 실시간으로 처리하지는 못하니까, function call을 추가함.
    - language model이 바로 처리하지 못하는 질문에는 지금 대답할 수 없다고 하고, 답변하기 위해 필요한 정보를 json에 담아서 반환
        - 예: 지금 몇 시야?, 수식 계산
            - 수식 계산 -> wolframalpha 에 넣으라고 설정할수 있음.
    - 내가 가지고 있는 함수의 목록을 만들고, language model이 작업을 할 때 이 함수 목록도 같이 보내준다.
    - 이 함수 목록에 해당하면 language model이 직접 만드는 것이 아니라, 이 함수를 실행해서 출력 결과를 보내주는 것
- Google AI Studio 팁!
    - 새 prompt 작성: [https://makersuite.google.com/app/prompts/new_freeform](https://makersuite.google.com/app/prompts/new_freeform)
    - 작성한 prompt 예시를 코드로 받을 수 있다.
        
        ![https://i.imgur.com/WChXCbr.png](https://i.imgur.com/WChXCbr.png)
        
        ![https://i.imgur.com/K3JU9ZO.png](https://i.imgur.com/K3JU9ZO.png)
        
    - if parts := msg.parts: 처럼 parts로 받아올거로 인식하고 있어서 저대로 넣으면 에러가 날 것
        - 고객센터에서 활용할 수 있음
            - 질문이 들어오면 지금 넣어놓은 것 중에서 제일 비슷한 걸 찾아서 리턴 (RAG사용, 생성 모델 x해서 더 안전하게)
        - 코드에서 수정한 부분 (branch - gemini)
            
            ```python
            # Initialize chat history
            if "messages" not in st.session_state:
              st.session_state.messages = [
                {
                  "role": "user",
                  "parts": ["Hello?"]
                },
                {
                  "role": "model",
                  "parts": ["Hi."]
                },
                {
                  "role": "user",
                  "parts": ["What can you do?"]
                },
                {
                  "role": "model",
                  "parts": ["Hi."]
                },
                {
                  "role": "user",
                  "parts": ["Who are you?"]
                },
                {
                  "role": "model",
                  "parts": ["Hi."]
                },
                {
                  "role": "user",
                  "parts": ["안녕?"]
                },
                {
                  "role": "model",
                  "parts": ["Hi."]
                },
                {
                  "role": "user",
                  "parts": ["Can you help me?"]
                },
                {
                  "role": "model",
                  "parts": ["Hi."]
                },
              ]
              st.session_state.init_msg_len = len(st.session_state.messages)
            ```
            
            ```python
            # Display messages in history for msg in st.session_state.messages[st.session_state.init_msg_len:]:   if parts := msg['parts']:     with st.chat_message('human' if msg['role'] == 'user' else 'ai'):       for p in parts:         st.write(p)
            ```
            
- Streamlit library import
    - `chatbot.py`
    - 컨셉: 입력창에 새로운 내용이 들어오면 파일을 처음부터 다시 실행
        
        ```python
        import streamlit as st
        
        # Title and Initialize chat history
        st.title("Echo Bot")
        
        # 다시 실행되었을 때 변수가 사라지지 않게 하기 위함
        if "messages" not in st.session state:
        	st.session_state.messages = []
        # Display chat messages from history on app rerun
        for message in st.session state.messages:
        	with st.chat_message(message["role"]): # role: AI or 유저
        		st.write(message["content"])
        
        ```
        
- React to user input
    - `chatbot.py` (continued)
        
        ```python
        if prompt := st.chat_input("What is up?"): # Display user message in chat message container
        	with st.chat_message("user"):
        		st.write(prompt)
        	# Add user message to chat history
        	st.session_state.messages.append({
        		"role": "user" ,
        		"content": prompt, })
        		response = f"(ECHO) {prompt}"
        		# Display assistant response in chat message container
        		with st.chat_message("assistant"): st.write(response) # Add assistant response to chat history
        			st.session_state.messages.append({
        			"role": "assistant" ,
        			"content": response, })
        
        ```