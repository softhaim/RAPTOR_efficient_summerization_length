import re
import json
import logging
import time
import random  # 파이썬 랜덤 모듈
import numpy as np  # numpy 추가
import torch  # torch 시드 고정
from tqdm import tqdm
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util

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

# 정답과 생성된 답변의 정규화 함수 (구두점 제거, 소문자 변환 등)
def normalize_answer(answer):
    answer = answer.lower().strip()
    
    # 특수 문자 및 불필요한 패턴 제거
    answer = re.sub(r'[.,:;!?]', '', answer)  # 구두점 제거
    answer = re.sub(r'\n', ' ', answer)  # 줄바꿈 문자 제거
    answer = re.sub(r'\t', ' ', answer)  # 탭 문자 제거
    answer = re.sub(r'\s+', ' ', answer)  # 연속된 공백 제거
    
    # '-' 문자 및 기타 특수 문자 처리
    answer = re.sub(r'-', ' ', answer)  # '-' 제거 또는 필요시 다른 방식으로 처리 가능
    
    # 연속된 '...' 패턴 제거
    answer = re.sub(r'\.{2,}', '', answer)  # 두 개 이상의 연속된 점을 제거
    
    return answer


# F1 스코어 계산 함수
def calculate_f1(predicted_answer, correct_answers):
    best_f1 = 0.0
    predicted_tokens = set(normalize_answer(predicted_answer).split())

    for correct_answer in correct_answers:
        correct_answer_tokens = set(normalize_answer(correct_answer).split())

        if predicted_tokens.issubset(correct_answer_tokens) or correct_answer_tokens.issubset(predicted_tokens):
            return 1.0  # 완전히 포함된 경우 F1 = 1.0 반환

        common_tokens = predicted_tokens.intersection(correct_answer_tokens)
        if len(common_tokens) == 0:
            continue

        precision = len(common_tokens) / len(predicted_tokens)
        recall = len(common_tokens) / len(correct_answer_tokens)
        f1_score = (2 * precision * recall) / (precision + recall)
        best_f1 = max(best_f1, f1_score)

    return best_f1


# 유사도 기반 평가 함수
def calculate_similarity(predicted_answer, correct_answers):
    similarity_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    generated_embedding = similarity_model.encode([predicted_answer])
    
    best_similarity = -1
    for correct_answer in correct_answers:
        correct_embedding = similarity_model.encode([correct_answer])
        similarity = util.pytorch_cos_sim(generated_embedding, correct_embedding).item()
        best_similarity = max(best_similarity, similarity)

    return best_similarity

# 불필요한 패턴 제거 및 줄바꿈, 마침표 처리 함수
def remove_unwanted_prefixes(generated_answer):
    patterns_to_remove = [
        r"^Here is the answer.*?:",
        r"^The final answer is:",  
        r"^Answer:",
        r"Generated response:",
        r"\$\\boxed{.*?}\$",   # 수학식 제거
        r"^The answer is:",
        r"^Step [0-9]+:",  
        r"##"  
    ]

    for pattern in patterns_to_remove:
        generated_answer = re.sub(pattern, '', generated_answer).strip()

    generated_answer = re.sub(r"\.\.+", "", generated_answer)  
    generated_answer = re.sub(r"\.\s*\.$", "", generated_answer)  
    generated_answer = re.sub(r"'\s*\.$", "", generated_answer)  
    generated_answer = re.sub(r"\n\n", "", generated_answer)  

    return generated_answer.strip()

# 프롬프트 생성 로직 수정
def adjust_prompt_with_example(question, correct_answers, answer_type):
    if question.strip().lower().endswith(('yes', 'no')):
        if answer_type == "short":
            return f"Q: {question}\nA: Answer with a short 'Yes' or 'No'."
        else:
            return f"Q: {question}\nA: Provide a detailed answer, but include 'Yes' or 'No' in the response."

    if answer_type == "short":
        return f"Q: {question}\nA: Provide a short and concise answer."
    elif answer_type == "detailed":
        return f"Q: {question}\nA: Provide a detailed answer in one sentences."
    return f"Q: {question}\nA: Provide a concise answer in one sentence."


def evaluate_summarization_length_variants(retrieval_augmented_model, split_name, base_output_path, length, top_k_list, final_output_path):
    set_seed(42)

    test_data = load_dataset("allenai/qasper", split=split_name)
    final_results = []

    for top_k in top_k_list:  # 고정된 top_k 값에 대해 반복
        logging.info(f"Evaluating with top_k: {top_k}, summarization length: {length}")

        # 시간 측정 시작
        start_time = time.time()

        results = []
        total_f1 = 0
        total_similarity = 0
        total_samples = 0

        for item in tqdm(test_data, desc=f"Evaluating with top_k {top_k} and summarization length {length}"):
            paper_id = item['id']
            full_text = item['full_text']['paragraphs']
            qas = item['qas']
            questions = qas['question']
            answers_list = qas['answers']
            question_ids = qas['question_id']

            retrieval_augmented_model.add_documents("\n".join([" ".join(section) for section in full_text]))

            for idx, question_text in enumerate(questions):
                question_id = question_ids[idx]
                answers_data = answers_list[idx]

                correct_answers = []
                for answer_entry in answers_data["answer"]:
                    free_form_answer = answer_entry.get('free_form_answer', "")
                    if free_form_answer:
                        correct_answers.append(free_form_answer)

                if not correct_answers:
                    continue

                if len(correct_answers[0].split()) <= 5:
                    answer_type = "short"
                elif len(correct_answers[0].split()) > 15:
                    answer_type = "detailed"
                else:
                    answer_type = "list"

                question_with_prompt = adjust_prompt_with_example(question_text, correct_answers, answer_type)

                try:
                    generated_answer, layer_info = retrieval_augmented_model.answer_question(
                        question=question_with_prompt,  
                        top_k=top_k,  # 고정된 top_k 사용
                        return_layer_information=True
                    )

                    generated_answer = remove_unwanted_prefixes(generated_answer)

                except Exception as e:
                    logging.error(f"Error while answering: {str(e)}")
                    continue

                f1 = calculate_f1(generated_answer, correct_answers)
                similarity = calculate_similarity(generated_answer, correct_answers)

                result = {
                    "paper_id": paper_id,
                    "question_id": question_id,
                    "question": question_text,
                    "correct_answers": correct_answers,
                    "generated_answer": generated_answer,
                    "f1_score": f1,
                    "similarity_score": similarity,
                    "layer_info": layer_info
                }

                results.append(result)
                total_f1 += f1
                total_similarity += similarity
                total_samples += 1

        avg_f1 = total_f1 / total_samples if total_samples > 0 else 0.0
        avg_similarity = total_similarity / total_samples if total_samples > 0 else 0.0

        # 시간 측정 종료
        end_time = time.time()
        elapsed_time = end_time - start_time

        logging.info(f"Average F1 Score with top_k {top_k}, summarization length {length}: {avg_f1:.4f}")
        logging.info(f"Average Similarity Score with top_k {top_k}, summarization length {length}: {avg_similarity:.4f}")
        logging.info(f"Evaluation time for top_k {top_k} and summarization length {length}: {elapsed_time:.2f} seconds")

        # 파일 덮어쓰지 않도록 고유한 이름으로 파일 저장
        output_path = f"{base_output_path}_topk_{top_k}_length_{length}.json"
        
        with open(output_path, 'w') as outfile:
            json.dump(results, outfile, ensure_ascii=False, indent=4)

        # 최종 결과를 리스트에 추가 (덮어쓰지 않음)
        final_results.append({
            "top_k": top_k,
            "summarization_length": length,
            "avg_f1": avg_f1,
            "avg_similarity": avg_similarity,
            "elapsed_time": elapsed_time
        })

    # 각 요약문 길이와 top_k 결과들을 최종 저장 (덮어쓰지 않음)
    with open(final_output_path, 'a') as final_outfile:  # 'a' 모드로 열어 기존 파일에 추가
        json.dump(final_results, final_outfile, ensure_ascii=False, indent=4)
        final_outfile.write("\n")  # 각 저장 결과 사이에 구분을 위해 줄바꿈 추가

if __name__ == "__main__":
    import argparse
    from QAModels import LLaMAQAModel
    from RetrievalAugmentation import RetrievalAugmentation, RetrievalAugmentationConfig
    from EmbeddingModels import SBertEmbeddingModel
    
    # 시드 고정
    set_seed(42)

    parser = argparse.ArgumentParser(description="QASPER Evaluation with Various Summarization Lengths and Total Tokens")
    parser.add_argument("--split_name", type=str, required=True, help="Data split to evaluate (e.g., 'test')")
    parser.add_argument("--base_output_path", type=str, required=True, help="Base path to save the evaluation results")
    parser.add_argument("--final_output_path", type=str, required=True, help="Path to save the final aggregated results")
    args = parser.parse_args()

    sbert_embedding_model = SBertEmbeddingModel(model_name="sentence-transformers/multi-qa-mpnet-base-cos-v1")
    llama_qa_model = LLaMAQAModel()

    summarization_lengths = [400]
    top_k_list = [25]

    for length in summarization_lengths:
        ra_config = RetrievalAugmentationConfig(
            qa_model=llama_qa_model,
            tr_embedding_model=sbert_embedding_model,
            tb_embedding_models={"LLaMA_EMB": sbert_embedding_model},
            tb_summarization_length=length
        )

        retrieval_augmented_model = RetrievalAugmentation(config=ra_config)

        evaluate_summarization_length_variants(retrieval_augmented_model, args.split_name, args.base_output_path, length, top_k_list, args.final_output_path)
