
# Summarization Length Expansion and Optimization for Enhanced Efficiency in Long-Context RAG Models


**RAPTOR**방법을 이용하여 요약문 최적화를 통해 더 효율적으로 문서를 재귀적인 트리 구조로 구성하여 정보 검색의 효율성과 문맥 인식을 향상시키는 새로운 접근 방식을 소개합니다. 이는 전통적인 언어 모델의 일반적인 한계를 해결합니다.

## 설치 방법

RAPTOR를 사용하기 전에 Python 3.8 이상이 설치되어 있는지 확인하세요. RAPTOR 저장소를 클론하고 필요한 의존성을 설치합니다:

```bash
git clone https://github.com/parthsarthi03/raptor.git
cd raptor
pip install -r requirements.txt
```

## 기본 사용법

RAPTOR를 시작하려면 다음 단계를 따르세요:

### RAPTOR 설정

먼저 OpenAI API 키를 설정하고 RAPTOR 구성을 초기화합니다:

```python
import os
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

from raptor import RetrievalAugmentation

# 기본 구성으로 초기화합니다. 고급 구성에 대해서는 문서를 확인하세요. [WIP]
RA = RetrievalAugmentation()
```

### 문서 추가

인덱싱을 위해 텍스트 문서를 RAPTOR에 추가합니다:

```python
with open('sample.txt', 'r') as file:
    text = file.read()
RA.add_documents(text)
```

### 질문에 답하기

인덱싱된 문서를 기반으로 질문에 답할 수 있습니다:

```python
question = "How did Cinderella reach her happy ending?"
answer = RA.answer_question(question=question)
print("Answer: ", answer)
```

### 트리 저장 및 로드

지정된 경로에 구성된 트리를 저장합니다:

```python
SAVE_PATH = "demo/cinderella"
RA.save(SAVE_PATH)
```

저장된 트리를 RAPTOR에 다시 로드합니다:

```python
RA = RetrievalAugmentation(tree=SAVE_PATH)
answer = RA.answer_question(question=question)
```

## QASPER 및 QuALITY 데이터셋 평가

RAPTOR는 다양한 요약문 크기와 top k 개수 조정을 통해 QASPER 및 QuALITY 데이터셋에 대한 평가를 실행할 수 있습니다. 아래 예시 코드와 함께 명령어를 사용하여 평가를 시작할 수 있습니다.

### 명령어 실행

QASPER 및 QuALITY 데이터셋 평가를 실행하기 위한 명령어 예시입니다:

```bash
python main.py --split_name test --base_output_path /path/to/base/output --final_output_path /path/to/final/output
```

이 명령어는 다음과 같은 인자들을 사용합니다:

- `--split_name`: 평가할 데이터셋의 분할 이름 (예: 'test')
- `--base_output_path`: 평가 결과를 저장할 기본 경로
- `--final_output_path`: 최종 집계 결과를 저장할 경로

### 코드 예시

```python
# 요약문 크기 및 top k 값을 조정하는 코드
evaluate_summarization_length_variants(retrieval_augmented_model, 'test', 'path/to/base/output', 400, [25], 'path/to/final/output')
```

## 라이센스

RAPTOR는 MIT 라이센스하에 출시됩니다. 자세한 내용은 RAPTOR 저장소를 참고하세요. (https://github.com/parthsarthi03/raptor.git)

## 인용

RAPTOR를 인용함에 따른 인용 표시

```bibtex
@inproceedings{sarthi2024raptor,
    title={RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval},
    author={Sarthi, Parth and Abdullah, Salman and Tuli, Aditi and Khanna, Shubh and Goldie, Anna and Manning, Christopher D.},
    booktitle={International Conference on Learning Representations (ICLR)},
    year={2024}
}
```
