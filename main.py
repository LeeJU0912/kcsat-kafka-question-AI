from confluent_kafka import Producer, Consumer, KafkaError

import os
import re
import time
import json
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pydantic import BaseModel

logging.basicConfig(level=logging.DEBUG)

consumerConfig = {
    'bootstrap.servers': 'kafka:9092',
    'group.id': 'HPCLab',
    'enable.auto.commit': False,
    'auto.offset.reset': 'latest'
}

producerConfig = {
    'bootstrap.servers': 'kafka:9092',
    'client.id': 'Question'
}

producer = Producer(producerConfig)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

llamaModel = AutoModelForCausalLM.from_pretrained(
    "/model/merged_llama-3-epoch-6-non-quantization-32-8_answer_new_prompt",
    device_map="auto")
llamaTokenizer = AutoTokenizer.from_pretrained("/model/merged_llama-3-epoch-6-non-quantization-32-8_answer_new_prompt")


class TextRequest(BaseModel):
    type: str
    definition: str
    mainText: str


# 러시아어(키릴 문자) 범위: \u0400-\u04FF
def contains_cyrillic(text):
    return bool(re.search(r'[\u0400-\u04FF]', text))


# 모든 한자 관련 유니코드 범위
def contains_hanzi(text):
    return bool(re.search(r'[\u4E00-\u9FFF]', text))


# 일본어 감지 함수 (히라가나, 가타카나, 한자 포함)
def contains_japanese(text):
    return bool(re.search(r'[\u3040-\u30FF\u4E00-\u9FFF]', text))


# 태국어 감지 함수
def contains_thai(text):
    return bool(re.search(r'[\u0E00-\u0E7F]', text))


def contains_vietnamese(text):
    # 베트남어에서 사용하는 특수 문자: â, ê, ô, ă, đ, ơ, ư 및 해당 성조 결합 기호를 포함한 범위 설정
    vietnamese_pattern = r"[ăâđêôơưĂÂĐÊÔƠƯ]"

    # 베트남어 문자와 일치하는지 확인
    return bool(re.search(vietnamese_pattern, text))


# 아랍어 감지 함수
def contains_arabic(text):
    return bool(re.search(r'[\u0600-\u06FF\u0750-\u077F]', text))


# 특수문자 감지 함수
def contains_special_characters(text):
    return bool(re.search(r'\uFFFD', text))


def generate_question_LLaMA(input_text, definition, model, tokenizer):
    # Tokenize input_text and definition
    prompt = f"""
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>

    {definition}<|eot_id|><|start_header_id|>user<|end_header_id|>

    {input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.to(model.device)

    # Generate question
    with torch.inference_mode():
        outputs = model.generate(input_ids=input_ids, max_new_tokens=1200,
                                 do_sample=True, top_p=0.9, temperature=0.9, pad_token_id=tokenizer.eos_token_id)

    generated_problem = outputs[0][input_ids.shape[-1]:]
    decoded_problem = tokenizer.decode(generated_problem, skip_special_tokens=True)
    return decoded_problem  # Return decoded text, not token IDs


def check_wrong_character(generated_problem):
    if contains_hanzi(generated_problem):
        logging.info("한자가 섞여 있습니다.")
        raise ValueError("한자가 섞여 있습니다.")
    if contains_cyrillic(generated_problem):
        logging.info("러시아어가 섞여 있습니다.")
        raise ValueError("러시아어가 섞여 있습니다.")
    if contains_vietnamese(generated_problem):
        logging.info("베트남어가 섞여 있습니다.")
        raise ValueError("베트남어가 섞여 있습니다.")
    if contains_arabic(generated_problem):
        logging.info("아랍어가 섞여 있습니다.")
        raise ValueError("아랍어가 섞여 있습니다.")
    if contains_thai(generated_problem):
        logging.info("태국어가 섞여 있습니다.")
        raise ValueError("태국어가 섞여 있습니다.")
    if contains_japanese(generated_problem):
        logging.info("일본어가 섞여 있습니다.")
        raise ValueError("일본어가 섞여 있습니다.")
    if contains_special_characters(generated_problem):
        logging.info("특수문자가 섞여 있습니다.")
        raise ValueError("특수문자가 섞여 있습니다.")


def special_filter_UNDERLINE(mainText: str):
    open_symbol_count = mainText.count("<U>")
    close_symbol_count = mainText.count("</U>")

    if open_symbol_count != 1 or close_symbol_count != 1:
        logging.info("<U></U> 토큰 파싱 오류")
        raise ValueError("<U></U> 토큰 파싱 오류")


def special_filter_BLANK(mainText):
    blank_symbol_count = mainText.count("(BLANK)")

    if blank_symbol_count != 1:
        logging.info("(BLANK) 토큰 파싱 오류")
        raise ValueError("(BLANK) 토큰 파싱 오류")


def special_filter_BLANK_AB(mainText):

    A_symbol_count = mainText.count("(A)")
    B_symbol_count = mainText.count("(B)")

    if A_symbol_count != 1 or B_symbol_count != 1:
        logging.info("(A), (B) 토큰 파싱 오류")
        raise ValueError("(A), (B) 토큰 파싱 오류")


def special_filter_GRAMMAR(mainText):
    for i in range(1, 6):
        if mainText.count(f"({i})") != 1:
            logging.info("문법 보기 매칭 오류")
            raise ValueError("문법 보기 매칭 오류")


def special_filter_SUMMARIZE_AB(mainText):
    open_symbol_count = mainText.count("<")
    close_symbol_count = mainText.count(">")

    if open_symbol_count != 1 or close_symbol_count != 1:
        logging.info("<, > 토큰 파싱 오류")
        raise ValueError("<, > 토큰 파싱 오류")

    A_symbol_count = mainText.count("(A)")
    B_symbol_count = mainText.count("(B)")

    if A_symbol_count != 1 or B_symbol_count != 1:
        logging.info("(A), (B) 토큰 파싱 오류")
        raise ValueError("(A), (B) 토큰 파싱 오류")


def special_filter_ORDERING(mainText):
    A_symbol_count = mainText.count("(A)")
    B_symbol_count = mainText.count("(B)")
    C_symbol_count = mainText.count("(C)")

    if A_symbol_count != 1 or B_symbol_count != 1 or C_symbol_count != 1:
        logging.info("(A), (B), (C) 토큰 파싱 오류")
        raise ValueError("(A), (B), (C) 토큰 파싱 오류")


def refine_output_to_json(questionType, generated_problem):
    data = {}

    logging.info(generated_problem)

    check_wrong_character(generated_problem)

    lines = generated_problem.split("\n")
    data.update({"title": str.strip(lines[0][4:])})
    data.update({"mainText": str.strip(lines[1][3:])})

    main_text = data.get("mainText")
    if main_text is not None:
        if questionType == "UNDERLINE":
            special_filter_UNDERLINE(main_text)
        elif questionType == "BLANK":
            special_filter_BLANK(main_text)
        elif questionType == "BLANK_AB":
            special_filter_BLANK_AB(main_text)
        elif questionType == "GRAMMAR":
            special_filter_GRAMMAR(main_text)
        elif questionType == "SUMMARIZE_AB":
            special_filter_SUMMARIZE_AB(main_text)
        elif questionType == "ORDERING":
            special_filter_ORDERING(main_text)

    selections = str.strip(lines[2][3:])

    selections = selections.replace('-', '－')

    selections = re.split(r'\(\d+\)', selections)

    selections = [selection.strip() for selection in selections]
    refined_selections = []
    for selection in selections:
        if selection != "":
            refined_selections.append(selection)

    choices = {"choices": []}

    for idx, selection in enumerate(refined_selections):
        choices["choices"].append(f"({idx + 1}) " + selection)

    data.update(choices)

    data.update({"answer": str.strip(lines[3][2:])})

    if not data.get("title") or not data.get("mainText") or not data.get("choices") or not data.get("answer"):
        if not isinstance(choices.get("choices"), list) or len(choices["choices"]) != 5:
            logging.info("파싱 오류")
            raise ValueError("값을 넣는 데에 오류가 있습니다.")

    return data


def make_question(request: TextRequest):
    questionType = request.type
    definition = request.definition
    mainText = request.mainText

    response = {}
    # 7번 시도
    for _ in range(7):
        logging.info("GET LLaMA Question")
        try:
            generated_problem = generate_question_LLaMA(mainText, definition, llamaModel, llamaTokenizer)

            refined_generated_problem = refine_output_to_json(questionType, generated_problem)

            response = refined_generated_problem

            break
        except Exception:
            logging.info("RETRY Create Question")
            continue

    return response


def returnMessage(key, data):
    producer.produce(topic='QuestionResponse',
                     key=key,
                     value=json.dumps(data, ensure_ascii=False).encode('utf-8'))
    producer.flush()


def consume_messages():
    consumer = Consumer(consumerConfig)

    consumer.subscribe(['QuestionRequest1'])

    logging.info("Load Complete...! ")

    try:
        while True:
            msg = consumer.poll(1.0)

            if msg is None:
                continue

            if msg.error():
                logging.info("Error")
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                else:
                    logging.info(msg.error())
            else:
                key = msg.key().decode('utf-8')
                message = msg.value().decode('utf-8')

                consumer.commit(msg)
                logging.info(message)

                # 객체 변환
                textRequest = TextRequest.model_validate_json(message)

                # 문제 생성
                response = make_question(textRequest)

                # 결과 반환
                returnMessage(key, response)
                logging.info("Return OK")


    except Exception as e:
        logging.info(e)
    finally:
        consumer.close()
        logging.info("Consumer closed")


def startup():
    logging.info("Starting consumer...")
    time.sleep(5)
    consume_messages()


if __name__ == "__main__":
    try:
        startup()
    except Exception as e:
        logging.info(e)
