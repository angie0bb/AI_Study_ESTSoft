# Python 튜토리얼

태그: Class, DataTypes, Decorator, Loop
No.: 1

## ⭐공식 문서로 틀 잡기 - Python

<aside>
💡 conda, ipynb 파일 생성해서 jupyter 환경에서 코드 예제 실행해보기

</aside>

[The Python Tutorial](https://docs.python.org/3/tutorial/index.html)

### Introduction

- Calculator
    - operators ☑️
        
        ```python
        # 연산 기호
        17 / 3 # classic division, return: float
        17 // 3 # floor division, return: minimum integer 
        17 % 3 # return: remainder of the division
        5 ** 2 # return: 5^2
        5 * 2 # return: 10
        5 * 2.0 # return: 10.0 -> full support for float! return float when mixed part exists
        5 + 2j # j: imaginary part for complex number
        ```
        
- Data Types
    - String ☑️
        
        #`multiple lines, escape, special characters, concatenation, index/indices/slicing`
        
        ```python
        # string: "" ''
        '1975' # digits and numerals with quotes = string
        type('1975') # check data type
        
        # multiple lines: """
        print("""
        1. first line
        2. second line
              """)
        ```
        
        ```python
        # escape: \ or use another type of quote in front of the object
        'doesn\'t' # return: doesn't
        "doesn't"
        
        # special characters
        # new line: \n, tab: \t
        s = "Hello!\nWorld!"
        s # return: Hello!\nWorld!
        print(s) # special characters 적용됨
        path = r"C:\Users\name" # \n escape 안하고 싶을 때
        print(path)
        ```
        
        ```python
        # string concatenate: + 
        3 * 'banana' # reutrn: 'bananabananabanana'
        'bana'+'na' # return: 'banana', literal 끼리는 + 생략 가능
        prefix = "Mr"
        prefix + ' banana'
        ```
        
        ```python
        # Index
        word = "abcdef"
        word[0]
        word[1]
        
        # Indices
        word[-1] # return: last character
        # -0 = 0이기 때문에 negative indice는 -1부터 시작함
        word[-2] # return: second last character 
        
        # Slicing: start is always included, end is always excluded
        word[0:2] # return: ab, [0+1,2) = 0<=x<2
        word[:2] # return: ab, [0+1:2)
        word[4:] # return: ef, 4+1번째 글자부터 끝까지 
        word[-2:] # return: ef, 뒤에서 두번째 글자부터 끝(마지막 글자)까지 
        
        word[:2] + word[2:] # return: ab + cdef = abcdef
        word[:4] + word[4:] # return: abcd + ef = abcdef
        ```
        
        ```python
        # Strings
        str = "abcde"
        print("str[2:-2] = ", str[2:-2]) #d: 2+1 번째 글자부터 뒤에서 두번째 글자까지인데, 두번째 글자는 미포함이니까..
        print("str[2:-1] = ", str[2:-1]) #cd
        ```
        
    - Lists ✅
        
        ```python
        lst = []
        
        for i in range(2,10):
            lst.append(i**2)
        print(lst)
        ```
        
        ```python
        # Data Structures
        
        # list
        language = ["a", "b", "c"]
        print(language[1:3])  # 1<=x<3 -> b, c
        
        language[2] = "d"
        print(language)
        
        language[1:3] = "e"
        print(language)
        ```
        
- Variable
    - assignment
        
        ```python
        # '=': to assign a value to a variable 
        width = 20
        height = 5 * 9 # 계산식도 가능
        width * height
        ```
        

### IF & Loop

- if ✅
    
    ```python
    # Control Flow
    # If, else statement 
    num = -1
    
    if num > 0:
      print("Positive")
    elif num == 0:
      print("Zero")
    else:
      print("Negative number")
    ```
    
    ```python
    a = "" # a 가 False
    if a: # a가 true 일때
      print(a)
    else:
      a += "a"
    print(a)
    ```
    
- range ✅
    
    ```python
    # 짝수 나열하기
    for i in range(0, 100, 2):
        print(i)
    
    for i in range(50):
        print(i*2+1)
    ```
    
- loop
    - for ☑️
        
        ```python
        # dictionary{}, list[], for loop
        # dictionary {key: value}
        name = input("학생의 이름을 입력해주세요.: ")
        exam_score = [{'a' : 60, 'b' : 80}, {'c' : 100, 'd' : 90}]
        
        for classes in exam_score:
            if name in classes: # 만약, b라는 학생이 반에 있으면:
                # concatenate 
                # print(name + "의 점수는 " + classes[name] + "점입니다.") -> +는 str끼리만 가능
                print(name + "의 점수는 " + str(classes[name]) + "점입니다.")   #b의 값(점수)을 출력하기
        ```
        
        ```python
        # 가장 높은 점수를 가지고 있는 학생 찾기
        exam_score = [{'a' : 60, 'b' : 80}, {'c' : 100, 'd' : 90}]
        maximum = 0
        highest_student = ""
        
        for classes in exam_score:
            for name in classes:
                if float(classes[name]) > maximum:
                    maximum = float(classes[name])
                    highest_student = name
        
        print("가장 높은 점수를 받은 학생은 " + highest_student + "입니다.")
        ```
        
        ```python
        # 반별 평균 점수 내기
        exam_score = [{'a' : 60, 'b' : 80}, {'c' : 100, 'd' : 90}]
        total = 0
        average = 0
        
        for classes in exam_score:  
            for name in classes:
                total += float(classes[name])
                average = round(total / len(classes), 2) # 2 decimals
            total = 0
            print(average)
        ```
        
        ```python
        # for loop 
        # generator를 따로 쓰지 않는 이상, 무한 루프가 발생하지 않음 
        numbers = [6,5,3,8,4,2] # iterable 한 것들 돌릴 수 있음 list, tuple 등
        
        sum = 0
        
        # iterate over the list
        for val in numbers:
            sum += val
        print(sum)
        ```
        
    - while & Errors, Exceptions ☑️
        
        [8. Errors and Exceptions](https://docs.python.org/3/tutorial/errors.html#handling-exceptions)
        
        - `while`: repeated execution as long as an expression is true
            - if the expression is `False` , else(if exists) is executed and the loop terminates
        - `try&except`
            
            ```python
            # Errors and Exceptions
            # 1) try clause가 실행됨. 만약 에러가 발생하지 않으면 while문은 break
            # 2) try clause가 실행됨. 만약 에러가 발생한다면 거기서 try clause는 멈추고, except clause 실행
            # 3) except에 명시된 에러면 except clause 실행됨. 
            # 4) 명시되지 않은 에러라면 except clause 실행 안 되고 try&break 이후 블록이 실행됨
            
            while True:
                try: 
                    choice = int(input("숫자를 입력하세요."))
                    print("성공")
                    break
                except ValueError: 
                    print("value error")
                print("Unknown")
            ```
            
            ```python
            # try & except 연습 2
            while True: #올바른 입력값을 받을때까지 반복
                try:
                    choice = input("일단...점심을 먹을까? [yes/no]")
                    allowed = ["yes", "no"]
                    if choice not in allowed:
                        raise ValueError # yes/no중에서 입력 받지 못하면 강제로 특정 에러 발생시키기
                    else:
                        print("성공")
                        break
                except ValueError:
                    print("[yes/no]중에서 입력해주세요.")
            ```
            
            ```python
            # while loop
            # 왠만하면 쓰지 말기 -> 따로 input을 받을일이 없으니까..
            
            n = 100
            
            # initialize
            sum = 0
            i = 1
            
            while i <=n:
              sum += i
              i += 1
            print(sum)
            ```
            
    - brake, continue, else ✅
        - `brake`: executed in the first suite(`while` clause), terminates the loop without executing the `else` clause.
        - `continue`: executed in the first suite, skips the rest and goes back to testing the expression
        
        ```python
        for i in "Mathematics":
            if i == "e":
                break
        else: # 꼭 이렇게 쓸 필요는 없음. 코드가 break되지 않고 끝까지 실행되었을때를 가정
            print("The end")
        
        print("break")
        ```
        
        ```python
        for val in "Mathematics":
            if val == "e":
                continue
            print(val)
        else:
            print("end")
        ```
        
        ```python
        a = []
        
        if not a:
            pass # 아무것도 안하기! class를 정의할때나, 
                 # 아직 구현할 건 없지만 해당 함수를 테스트해야할 떄
            # raise NotImplementedError # give notion 
        else:
            print(a)
        ```
        
        ```python
        for i in "Mathematics":
            pass
        else:
            print("end")
        ```
        

### Functions

- defining functions, decorator ✅
    
    [Python Decorators (With Examples)](https://www.programiz.com/python-programming/decorator)
    
    ```python
    # Function
    def print_lines(): # (인자로 받을 것 넣기), 아무것도 받지 않아도 됨
        print("I am line 1.")
        print("I am line 2.")
    
    print_lines()
    
    def add_numbers(a,b):
        sum = a + b
        return sum
    
    print(add_numbers(2,3))
    ```
    
    ```python
    # Decorator **: a function that returns a (modified) function
    
    def one_more(ftn):
        def wrapper(a, b): # 함수 안에서 함수를 정의. a와 b를 ftn이란 함수에 넣고 1을 더해서 return 
            print(ftn(a,b)+1)
            return ftn(a,b) + 1
        return wrapper
    
    one_more(add_numbers)(4,5)
    ```
    
    ```python
    @one_more  #one more 이라는 함수에 add_numbers_one_more을 집어넣고 이름을 바꿔줌
    def add_numbers_one_more(a,b):
        sum = a+b
        return sum
    
    add_numbers_one_more(4,5)
    ```
    
    ```python
    def more(n): # n이라는 인자를 받아서 decorator outer를 return하는 함수
        def outer(ftn):
            def inner(a,b):
                return ftn(a,b) + n
            return inner
        return outer
    
    more(10)(add_numbers)(4,5) #
    ```
    
    ```python
    @more(10)
    def add_numbers_ten_more(a,b):
        sum = a+b 
        return sum
    
    add_numbers_ten_more(4,5)
    ```
    
- lambda ✅
    
    ```python
    # Lambda Function: 편의 기능
    
    def square(x):
        return x ** 2
    
    print(square(5))
    
    square = lambda x: x**2 #함수의 인자를 다른 함수로 넘겨야할때 사용
    print(square(5))
    ```
    
    ```python
    numbers = [6,5,3,8,4,2]
    # 각각의 원소를 제곱하고 싶다면?
    
    numbers2 = map(lambda x: x ** 2, numbers) # map은 리스트를 전부 만들어주지는 않음.
    print(numbers2)
    list(numbers2)
    ```
    
    ```python
    numbers_lt_5 = filter(lambda x: x<5, numbers) #함수가 참을 리턴할때만 가지고 옴. 
    print(list(numbers_lt_5))
    
    # res = []
    # for i in numbers:
    #     if i < 5:
    #         res.append(i)
    # res
    ```
    
- generator ✅
    - generator 란?
        
        ```python
        # Generators
        
        def my_gen():
            n = 1
            print("first print")
            yield n
            #function의 return과 유사함. 다만, 함수는 return을 딱 한번만 하고, 거기서 함수가 끝남. return 밑은 실행 안 됨
            #yield는 거기서 실행을 끝내지 않음.
        
            n += 1
            print("second print")
            yield n 
        
            n += 1
            print("last print")
            yield n
        
        # it returns an object but does not execute immediately
        a = my_gen()
        print(next(a)) # next를 통해서 값을 받아옴
        ```
        
        ```python
        # using for loop
        
        # generator는 next를 통해서 값을 호출할때만 값을 계산하기 때문에 map 자체는 빠르게 리턴됨. 
        # loop를 돌때만 값을 계산
        for item in my_gen(): #stop iteration이 raise 될때까지 실행
            # range()도 generator의 일종
            print(item)
            # break
        else: #stop iteration이 raise 되면 else로 들어감
            print("End")
        ```
        
        ```python
        # 무한루프 generator
        def bad_generator():
            while True:
                yield 1
        
        # for i in bad_generator():
        #     print(i)
        ```
        
    - `range()` ✅
        
        ```python
        # range()
        print(range(1,10))
        
        numbers = range(1,10) # 1<=x<10
        print(list(numbers)) # 1~9
        print(tuple(numbers))
        print(set(numbers))
        
        # step size
        numbers1 = range(1, 10, 2)
        print(list(numbers1))
        
        numbers2 = range(10, 1, -2)
        print(list(numbers2))
        ```
        
    - 231027 조별 예제 코드 - 랜덤 발표자 뽑기 generator 활용
        
        ```python
        from random import randrange
        
        names = ["한상준", "김선들", "송승민", "안지우"]
        
        def generator(n):
          for i in range(n):
            # 발표 할 발표자를 랜덤으로 반환 및 삭제
            selected = names.pop(randrange(len(names)))
            print(f"{i+1}번 발표자는 {selected} 입니다.")
            yield
        
        lets_pick_one = generator(5)
        
        #---------------------------------------------------------
        try:
          next(lets_pick_one)
        except (StopIteration, ValueError):
          print("The end")
        
        # 데코레이터 부분을 제거하고 제너레이터만 사용해서 간소화 했습니다.
        ```
        
    

### Data Structures (More on Introduction)

- boolean ✅
    
    ```python
    # Boolean Types
    print(type(True))
    
    print(True and True) # True
    print(True and False) # False
    print(True or False) # True
    print(False or False) # False
    ```
    
    ```python
    # empty list의 bool값 = False
    bool([])
    
    a = []
    if not bool(a):
        a.append(0.0)
    else:
        print(a)
    ```
    
- Lists
    - Stacks
    - Queues
    - Nested List
    - List Comprehension ✅
        
        ```python
        # list comprehension
        numbers = [6,5,3,8,4,2]
        print([x**2 for x in numbers]) #[]주의, ()로 하면 map와 동일하게 generator로 처리됨
        
        for i in (x**2 for x in numbers): # generator는 next로 호출을 해야 불러오니까 -> for loop에서 for가 해줌
            print(i)
        ```
        
- Tuples, Sequences ✅
    
    ```python
    # tuple: lists are mutable but tuples are not immutable
    language = ("a", "b", "c")
    type(language)
    
    language[2] = "d" # error
    ```
    
- Sets ✅
    
    ```python
    # Sets
    my_set = set() # empty set
    my_set = {1,2,"hello",(3,2)}
    type(my_set) # set
    
    1 in my_set # True
    print(my_set)
    
    my_set.add(5) # 순서?
    print(my_set) 
    
    my_set.update([2,3,4]) # 중복?
    print(my_set)
    ```
    
    ```python
    # 합집합, 교집합
    A = {1,2,3}
    B = {2,3,4,5}
    
    print(A | B)
    print(A & B)
    print(A - B)
    print(A ^ B)
    ```
    
- Dictionaries ✅
    
    ```python
    # Dictionary
    my_dict = {} # = dict()
    my_dict = {"key":"value", "key2":"value2"}
    print(my_dict)
    
    my_dict["key"]
    
    del my_dict["key2"]
    my_dict
    ```
    

### Modules

- Modules
- Packages

### Input & Output

- Output formatting
- Reading and Writing Files

### Classes

- Class ✅
    - Class: 모델 파라미터를 저장하고, 필요한 연산도 같이 저장해두는 역할로 쓰임
        
        ```python
        class MyClass(): # class MyClass(object)라고 써도 됨
            def __init__(self, a): # Class 초기값 미리 넣어주기, 클래스가 처음 생성될때 호출됨
                self.a = a
            def add(self, b):
                self.a = self.a + b
                return self
        
        obj1 = MyClass(10) # obj1이 self가 됨
        print(obj1.a)
        
        obj2 = MyClass(15)
        obj2.a
        ```
        
    - Class 상속
        
        ```python
        # 상속
        class MyClass2(MyClass): # 아들 클래스(부모 클래스):
            def mul(self, b):
                self.a = self.a * b
                return self
            
        obj3 = MyClass2(10)
        print(obj3.a)
        
        obj3.mul(3)
        print(obj3.a)
        
        # object-oriented 사용법?
        obj3.add(5).mul(2).add(3)
        ```
        
    - Annotation: 공동 작업 시에 유의하기
        
        ```python
        class MyClass():
            a: float # a의 타입 지정, 일반 함수에서도 파라미터의 타입(데이터 형, 어떤 클래스의 object 등)을 지정해줄 수 있음
            # 근데 강제하는 건 아님.... 딴 거 넣어도 잘 돌아감
            def __init__(self, a:float):
                self.a = a
            def add(self, b:float):
                self.a += b
                return self
            
        obj = MyClass(3.0)
        print(obj.a)
        obj.add(2.0).add(3.0)
        print(obj.a)
        
        # Annotation is NOT static typing
        obj = MyClass(3)
        print(obj.a)
        obj.add(2).add(3)
        print(obj.a)
        ```
        

### Library

###