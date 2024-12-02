import logging
import json
import re
import random
import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import time
from bs4 import BeautifulSoup  # BeautifulSoup 추가

# 시드 고정 함수
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic 모드 설정
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# JSON 파일 전처리 및 year 컬럼 타입 수정
def preprocess_json_file(file_path):
    try:
        data = pd.read_json(file_path, lines=True)
        if 'year' in data.columns:
            data['year'] = data['year'].astype(str)
        data.to_json(file_path, orient='records', lines=True)
        logging.info("JSON 파일이 성공적으로 수정되었습니다.")
    except ValueError as e:
        logging.error(f"파일을 로드하는 중 오류 발생: {e}")
        raise e

# HTML을 텍스트로 변환하는 함수
def html_to_text(html_content):
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        return soup.get_text(separator="\n")
    except Exception as e:
        logging.error(f"HTML to text 변환 중 오류 발생: {e}")
        return ""

# 문서 본문에서 불필요한 공백 제거 및 특수 문자 처리
def clean_article_text(article_text):
    article_text = article_text.replace('\\"', '"')  # 이스케이프된 따옴표 복원
    article_text = re.sub(r'\\n', ' ', article_text)  # 불필요한 줄바꿈 제거
    article_text = re.sub(r'\s+', ' ', article_text).strip()  # 연속된 공백 제거

    # HTML 형식의 본문을 텍스트로 변환
    if "<html" in article_text.lower():
        logging.info("HTML 본문이 감지되어 텍스트로 변환합니다.")
        article_text = html_to_text(article_text)
    
    return article_text

# 정규화 함수
def normalize_answer(answer):
    answer = answer.lower().strip()
    answer = re.sub(r'[.,:;!?]', '', answer)
    answer = re.sub(r'\n', ' ', answer)
    answer = re.sub(r'\t', ' ', answer)
    answer = re.sub(r'\s+', ' ', answer)
    return answer

# F1 스코어 계산 함수
def calculate_f1(predicted_answer, correct_answers):
    best_f1 = 0.0
    predicted_tokens = set(normalize_answer(predicted_answer).split())

    for correct_answer in correct_answers:
        correct_answer_tokens = set(normalize_answer(correct_answer).split())

        # 정답 또는 예측 답변이 완전히 포함된 경우 F1 = 1.0 반환
        if predicted_tokens.issubset(correct_answer_tokens) or correct_answer_tokens.issubset(predicted_tokens):
            return 1.0

        # 공통 토큰 계산
        common_tokens = predicted_tokens.intersection(correct_answer_tokens)
        if len(common_tokens) == 0:
            continue

        # Precision 및 Recall 계산
        precision = len(common_tokens) / len(predicted_tokens)
        recall = len(common_tokens) / len(correct_answer_tokens)

        # F1 스코어 계산
        f1_score = (2 * precision * recall) / (precision + recall)
        best_f1 = max(best_f1, f1_score)

    return best_f1

# 유사도 계산 함수
def calculate_similarity(predicted_answer, correct_answers, similarity_model):
    generated_embedding = similarity_model.encode([predicted_answer])
    best_similarity = -1
    for correct_answer in correct_answers:
        correct_embedding = similarity_model.encode([correct_answer])
        similarity = util.pytorch_cos_sim(generated_embedding, correct_embedding).item()
        best_similarity = max(best_similarity, similarity)
    return best_similarity

# 불필요한 패턴 제거
def remove_unwanted_prefixes(generated_answer):
    patterns_to_remove = [
        r"^Answer:",  
        r"Generated response:",
        r"\$\\boxed{.*?}\$",  
        r"^The answer is:",
        r"^Step [0-9]+:" 
    ]
    for pattern in patterns_to_remove:
        generated_answer = re.sub(pattern, '', generated_answer).strip()
    return generated_answer

# 프롬프트 생성 로직
def adjust_prompt_with_example(question, correct_answers):
    # Yes/No 질문 처리
    if question.strip().lower().endswith(('yes', 'no')):
        return f"Q: {question}\nA: Answer with a short 'Yes' or 'No'."

    # 정답의 길이에 따라 답변 길이 요청
    if len(correct_answers[0].split()) <= 5:
        return f"Q: {question}\nA: Provide a short and concise answer."
    elif len(correct_answers[0].split()) > 15:
        return f"Q: {question}\nA: Provide a detailed answer in one or two sentences."
    
    return f"Q: {question}\nA: Provide a concise answer."

# 평가 함수
def evaluate_quality(retrieval_augmented_model, data_path, base_output_path, length, top_k_list, final_output_path, seed=42, test_samples=2):
    set_seed(seed)  # 시드 고정 함수 호출
    
    similarity_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    
    preprocess_json_file(data_path)
    dataset = load_dataset('json', data_files=data_path)['train'].select(range(test_samples))

    final_results = []

    for top_k in top_k_list:
        logging.info(f"Evaluating with top_k: {top_k}, summarization length: {length}")

        total_f1 = 0
        total_similarity = 0
        total_samples = 0
        results = []

        start_time = time.time()

        for item in tqdm(dataset, desc=f"Evaluating with top_k {top_k} and summarization length {length}"):
            article_id = item['article_id']
            article_text = clean_article_text(item['article'])  # 문서 본문 처리
            questions = item['questions']

            retrieval_augmented_model.add_documents(article_text)

            for question_data in questions:
                question_text = question_data['question']
                options = question_data['options']
                correct_answers = options

                # 프롬프트 조정
                question_with_prompt = adjust_prompt_with_example(question_text, correct_answers)

                try:
                    generated_answer = retrieval_augmented_model.answer_question(
                        question=question_with_prompt,
                        top_k=top_k
                    )
                    generated_answer = remove_unwanted_prefixes(generated_answer)

                except Exception as e:
                    logging.error(f"Error while answering: {str(e)}")
                    continue

                f1 = calculate_f1(generated_answer, correct_answers)
                similarity = calculate_similarity(generated_answer, correct_answers, similarity_model)

                result = {
                    "article_id": article_id,
                    "question": question_text,
                    "correct_answers": correct_answers,
                    "generated_answer": generated_answer,
                    "f1_score": f1,
                    "similarity_score": similarity
                }

                results.append(result)
                total_f1 += f1
                total_similarity += similarity
                total_samples += 1

        avg_f1 = total_f1 / total_samples if total_samples > 0 else 0.0
        avg_similarity = total_similarity / total_samples if total_samples > 0 else 0.0

        end_time = time.time()
        elapsed_time = end_time - start_time

        output_path = f"{base_output_path}_topk_{top_k}_length_{length}.json"
        with open(output_path, 'w') as outfile:
            json.dump(results, outfile, ensure_ascii=False, indent=4)

        logging.info(f"Evaluation time for top_k {top_k}, summarization length {length}: {elapsed_time:.2f} seconds")

        final_results.append({
            "top_k": top_k,
            "summarization_length": length,
            "avg_f1": avg_f1,
            "avg_similarity": avg_similarity,
            "evaluation_time_sec": elapsed_time
        })

    with open(final_output_path, 'a') as final_outfile:
        json.dump(final_results, final_outfile, ensure_ascii=False, indent=4)
        final_outfile.write("\n")


if __name__ == "__main__":
    import argparse
    from QAModels import LLaMAQAModel
    from RetrievalAugmentation import RetrievalAugmentation, RetrievalAugmentationConfig
    from EmbeddingModels import SBertEmbeddingModel

    parser = argparse.ArgumentParser(description="QuALITY Evaluation with Summarization Lengths and top_k")
    parser.add_argument("--data_path", type=str, required=True, help="Path to QuALITY data")
    parser.add_argument("--base_output_path", type=str, required=True, help="Base path to save individual evaluation results")
    parser.add_argument("--final_output_path", type=str, required=True, help="Path to save final aggregated results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--test_samples", type=int, default=2, help="Number of test samples to evaluate")
    args = parser.parse_args()
    
    # 시드 고정
    set_seed(args.seed)

    sbert_embedding_model = SBertEmbeddingModel(model_name="sentence-transformers/multi-qa-mpnet-base-cos-v1")
    llama_qa_model = LLaMAQAModel()

    summarization_lengths = [100, 150, 200, 250, 300, 400, 500]
    top_k_list = [10, 15, 20, 25]

    for length in summarization_lengths:
        ra_config = RetrievalAugmentationConfig(
            qa_model=llama_qa_model,
            tr_embedding_model=sbert_embedding_model,
            tb_embedding_models={"LLaMA_EMB": sbert_embedding_model},
            tb_summarization_length=length
        )

        retrieval_augmented_model = RetrievalAugmentation(config=ra_config)

        evaluate_quality(
            retrieval_augmented_model, 
            args.data_path, 
            args.base_output_path, 
            length, 
            top_k_list, 
            args.final_output_path, 
            seed=args.seed,
            test_samples=args.test_samples
        )
