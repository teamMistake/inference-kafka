import os
import json
from kafka import KafkaConsumer, KafkaProducer
from services import JamoService
from transformers import AutoTokenizer
import torch
import wget
import typing as T
import signal
import time
# import intel_extension_for_pytorch as ipex

torch.set_grad_enabled(False)
torch.set_float32_matmul_precision("high")

class Jamo():
    def __init__(self):
        self.jamo = JamoService()
        self.tokenizer = AutoTokenizer.from_pretrained("hg_tokenizer")
        
        self.SOS_TOKEN = "<s>"
        self.EOS_TOKEN = "</s>"

    def generate(
        self,
        prompts: T.List[str],
        max_token:int,
        temperature: float, 
        top_k: int
    ):
        multibatch = len(prompts) != 1

        if not multibatch:
            parsed_prompt = Jamo.parsing_prompt(prompts[0])
            parsed_prompt = f"{self.SOS_TOKEN} {parsed_prompt}"
            prompt_idx = self.encode(parsed_prompt).squeeze(0)
            # kwgs = {"idx":prompt_idx, "max_token":max_token}

            predicted_idx = self.jamo.generate_idx(prompt_idx, max_token=max_token, temperature=temperature, top_k=top_k)
            predicted_text = self.decode(predicted_idx)
            answer = [Jamo.clean_response(predicted_text)]
        else:
            parsed_prompts = [f"{self.SOS_TOKEN} {Jamo.parsing_prompt(p)}" for p in prompts]
            prompt_idx = [self.tokenizer.encode(parsed_prompt) for parsed_prompt in parsed_prompts]
            max_length = max([len(i) for i in prompt_idx])
            prompt_idx = [[1]*(max_length-len(idx))+idx for idx in prompt_idx]
            prompt_idx = torch.LongTensor(prompt_idx)

            predicted_idx, finish_idxs = self.jamo.multibatch_generate(prompt_idx, max_token=max_token, temperature=temperature, top_k=top_k)

            predicted_idx = [l[:int(i)] for l, i in zip(predicted_idx.tolist(), finish_idxs)]
            predicted_texts = [self.decode(idx) for idx in predicted_idx]
            answer = [Jamo.clean_response(a) for a in predicted_texts]
            
        return answer

    def generator(self, prompts: str, max_token: int, temperature: float, top_k: int):
        multibatch = len(prompts) != 1
        
        if not multibatch:
            parsed_prompt = Jamo.parsing_prompt(prompts[0])
            parsed_prompt = f"{self.SOS_TOKEN} {parsed_prompt}"
            cur = len(parsed_prompt)

            prompt_idx = self.encode(parsed_prompt).squeeze(0)
            # kwgs = {"idx":prompt_idx, "max_token":max_token}

            full_answer = ""

            try:
                tmp_answer = ""
                for predicted_idx in self.jamo.streaming_generate_idx(prompt_idx, max_token=max_token, temperature=temperature, top_k=top_k):
                    if predicted_idx == None:
                        full_answer = Jamo.clean_response(tmp_answer)
                        raise StopIteration
                    
                    target = self.decode(predicted_idx)
                    target = target[:-1]
                    tmp_answer = target
                    
                    new = target[cur:]
                    cur = len(target)
                    yield [new], [False]

            except StopIteration:
                yield [full_answer], [True]

        else: 
            parsed_prompts = [f"{self.SOS_TOKEN} {Jamo.parsing_prompt(p)}" for p in prompts]
            prompt_idxes = [self.tokenizer.encode(parsed_prompt) for parsed_prompt in parsed_prompts]
            max_length = max([len(i) for i in prompt_idxes])
            prompt_idxes = [[1]*(max_length-len(idx))+idx for idx in prompt_idxes]

            # return eos
            eos = [False] * len(prompts)
            cur = [len(self.decode(prompt_idx)) for prompt_idx in prompt_idxes]

            full_answers = []

            prompt_idxes = torch.LongTensor(prompt_idxes)

            try: 
                tmp_answers = []
                for predicted_idx, finish_idxs in self.jamo.multibatch_streaming(prompt_idxes, max_token=max_token, temperature=temperature, top_k=top_k): 
                    if predicted_idx == None:
                        full_answers = [Jamo.clean_response(a) for a in predicted_texts]
                        raise StopIteration      
                    
                    eos = [finish_idx != 0 for finish_idx in finish_idxs]

                    tmp_finish_idxs = [f if f != 0 else len(predicted_idx[0]) for f in finish_idxs ]
                    roi_predicted_idx = [l[:int(i)] for l, i in zip(predicted_idx.tolist(), tmp_finish_idxs)]
                    predicted_texts = [self.decode(idx)[:-1] for idx in roi_predicted_idx]
                    tmp_answers = predicted_texts
                    new = [a[c:] for a, c in zip(tmp_answers,cur)]
                    cur = [len(target) for target in predicted_texts]
                    new = [Jamo.clean_response(predicted_texts[i]) if e else n for i, (e, n) in enumerate(zip(eos, new))]

                    yield new, eos
                    
            except StopIteration:
                yield full_answers, [True] * len(prompts)

        return
    

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
    def parsing_prompt(instruction):
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
print("Load the model")


print("?")
with open("/tmp/healthy", "w") as f:
    f.write("I'm GOOD, Definitely!")
print("??")
# model = ipex.optimize(model)

# prompts = ["너는 누구니?", "안녕 반가워, 내 이름은 윤승현이라고 해.", "너가 가장 좋아하는 음식은?"]
# for i, eos in model.generator(prompts, 100, 0.8, 10):
#     print(i, eos)

def timeout_handler(signum, frame):
    raise TimeoutError("Timeout occurred.")

def set_timeout(seconds):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)

while True:
    # GET requests
    req_ids = []
    reqs = []
    stream = None
    max_token = None
    temperature = None
    top_k = None

    # try:
        # set_timeout(2)

    print("Get Requests")
    start = time.time()

    target_time = 1000
    poll_time = 50

    # for msg in consumer:
    

    print("Generation Start")
    if stream:
        print("Streaming Creation")
        seq = 1
        already = list(range(len(reqs)))
        done = [0] * len(already)
        total = {key:"" for key in already}
        for (parts, eoses) in model.generator(req, max_token=max_token, temperature=temperature, top_k=top_k):
            seq += 1
            print(parts, eoses)
            for target_index in already:
                if done[target_index] != 1:
                    part, eos = parts[target_index], eoses[target_index]
                    if eos:
                        total[target_index] = part
                        done[target_index] = 1
                        # already.remove(target_index)
                    else:
                        total[target_index] += part
                    
                    producer.send(RESP_TOPIC, json.dumps({
                        'resp_partial': part,
                        'resp_full': total[target_index],
                        'eos': eos
                    }).encode(), headers=[("req_id", req_ids[target_index].encode()), ("seq_id", str(seq).encode())])
    else:   
        respThinks = model.generate(reqs, max_token=max_token, temperature=temperature, top_k=top_k)

        for respThink, req_id in zip(respThinks, req_ids):
            producer.send(RESP_TOPIC, json.dumps({
                'resp_partial': respThink,
                'resp_full': respThink,
                'eos': True
            }).encode(), headers=[("req_id", req_id.encode()), ("seq_id", b'1')])
