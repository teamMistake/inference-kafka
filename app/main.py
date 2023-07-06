import os
import json
from kafka import KafkaConsumer, KafkaProducer
from services import JamoService
from transformers import AutoTokenizer
import torch

torch.set_grad_enabled(False)
torch.set_float32_matmul_precision("high")

bootstrap = os.environ["KAFKA_BOOTSTRAP_ADDRESS"]
req = os.environ["KAFKA_REQ_TOPIC"]
resp = os.environ["KAFKA_RESP_TOPIC"]
username = os.environ["KAFKA_USERNAME"]
password = os.environ["KAFKA_PASSWORD"]
node_id = os.environ["NODE_ID"]

model_name = os.environ["MODEL_NAME"]

consumer = KafkaConsumer(req, group_id=f'inf-{model_name}',
                         client_id=f'inference-cons-{node_id}',
                         bootstrap_servers=bootstrap,
                         security_protocol="SASL_PLAINTEXT",
                         sasl_mechanism="SCRAM-SHA-512",
                         sasl_plain_username=username,
                         sasl_plain_password=password)
producer = KafkaProducer(bootstrap_servers=bootstrap,
                         client_id=f'inference-prod-{node_id}',
                         security_protocol="SASL_PLAINTEXT",
                         sasl_mechanism="SCRAM-SHA-512",
                         sasl_plain_username=username,
                         sasl_plain_password=password)

class Jamo():
    def __init__(self):
        self.jamo = JamoService()
        self.tokenizer = AutoTokenizer.from_pretrained("hg_tokenizer")
        
        self.SOS_TOKEN = "<s>"
        self.EOS_TOKEN = "</s>"

    def generate(
        self,
        prompt:str,
        max_token:int
    ):
        parsed_prompt = Jamo.parsing_prmopt(prompt)
        parsed_prompt = f"{self.SOS_TOKEN} {parsed_prompt}"

        prompt_idx = self.encode(parsed_prompt).squeeze(0)
        # kwgs = {"idx":prompt_idx, "max_token":max_token}

        predicted_idx = self.jamo.generate_idx(prompt_idx, max_token=max_token)
        predicted_text = self.decode(predicted_idx)
        answer = Jamo.clean_response(predicted_text)

        return answer

    def generator(self, prompt: str, max_token: int):
        parsed_prompt = Jamo.parsing_prmopt(prompt)
        parsed_prompt = f"{self.SOS_TOKEN} {parsed_prompt}"
        cur = len(parsed_prompt)

        prompt_idx = self.encode(parsed_prompt).squeeze(0)
        # kwgs = {"idx":prompt_idx, "max_token":max_token}

        full_answer = ""

        try:
            for predicted_idx in self.jamo.streaming_generate_idx(prompt_idx, max_token=max_token):
                target = self.decode(predicted_idx)
                if predicted_idx == None:
                    full_answer = Jamo.clean_response(target)
                    raise StopIteration
                target = target[:-1]
                new = target[cur:]
                cur = len(target)
                yield new, False

        except StopIteration:
            return full_answer, True

    def encode(self, text) -> torch.Tensor:
        try:
            token = self.tokenizer.encode(text, return_tensors="pt")    
        except Exception as e:
            raise e
        return token
    
    def decode(self, idx) -> str:
        try:
            text = self.tokenizer.decode(idx)    
        except Exception as e:
            raise e
        return text

    @staticmethod
    def parsing_prmopt(instruction):
        chat_parser = (
            "명령어에 따른 요청을 적절히 완료하는 응답을 작성하세요.\n\n"
            "### 명령어:\n{instruction}\n\n### 응답:\n"
        )

        parsed_prompt = chat_parser.format_map({"instruction":instruction})

        return parsed_prompt

    @staticmethod
    def clean_response(answer):
        splitted = answer.strip().split("### 응답:")
        cleaned_answer = splitted[-1].rstrip().strip() if len(splitted) > 1 else ""

        return cleaned_answer

model = Jamo()

for msg in consumer:
    headerList = msg.headers
    model_match = False
    req_id = ''
    for k, v in headerList:
        if k == "target_model":
            if v.decode() == model_name:
                model_match = True
        if k == "req_id":
            req_id = v.decode()
    if not model_match:
        continue

    content = msg.value.decode()

    parsed = json.loads(content)
    req = parsed['req']
    max_token = parsed["max_token"]

    if parsed['stream']:
        seq = 0
        total = ""
        for part, eos in model.generator(req, max_token=max_token):
            seq += 1
            if eos:
                total = part
            else: 
                total += part
            producer.send(resp, json.dumps({
                'resp_partial': part,
                'resp_full': total,
                'eos': eos
            }).encode(), headers=[("req_id", req_id.encode()), ("seq_id", str(seq).encode())])
    else:   
        respThink = model.generate(req, max_token=max_token)
        producer.send(resp, json.dumps({
            'resp_partial': respThink,
            'resp_full': respThink,
            'eos': True
        }).encode(), headers=[("req_id", req_id.encode()), ("seq_id", b'1')])


