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

bootstrap = os.environ["KAFKA_BOOTSTRAP_ADDRESS"]
req = os.environ["KAFKA_REQ_TOPIC"]
RESP_TOPIC = os.environ["KAFKA_RESP_TOPIC"]
username = os.environ["KAFKA_USERNAME"]
password = os.environ["KAFKA_PASSWORD"]
node_id = os.environ["NODE_ID"]

model_name = os.environ["MODEL_NAME"]
model_url = os.environ["MODEL_URL"]
max_batch_item = int(os.environ["MAX_BATCH_SIZE"])

timeout_ms = os.environ["TIMEOUT_MS"]
IS_CUDA = os.environ["IS_CUDA"]

print("model download start")
start = time.time()
wget.download(model_url, out="model_store/jamo.tar")
print("model download success in", time.time()-start)
# target_file = glob.glob("model_store/*.tar")[0]
# new_filename = "model_store/jamo.tar"
# os.rename(target_file, new_filename)

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
        self.device = "cuda" if IS_CUDA else "cpu"
        self.jamo = JamoService(self.device)
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

class GracefulKiller:
  kill_now = False
  def __init__(self):
    signal.signal(signal.SIGINT, self.exit_gracefully)
    signal.signal(signal.SIGTERM, self.exit_gracefully)

  def exit_gracefully(self, *args):
    self.kill_now = True

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

killer = GracefulKiller()
while not killer.kill_now:
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
    for _ in range(int(target_time//poll_time)):
        if len(req_ids) >= max_batch_item or time.time() - start > target_time/1000:
            print("timeout!")
            break
        records = consumer.poll(50, max_records=1)
        if records is None or records == {}: continue

        for i in records.items():
            msg = i[1][0]
            break

        print(msg)

#         ConsumerRecord(topic='model-req', partition=6, offset=377, timestamp=1690111922558, timestamp_type=0, key=None, value=b'{"req":"\xec\x95\x88\xeb\x85\x95","context":[{"type":"HUMAN","message":"\xec\x95\x88\xeb\x85\x95"},{"type":"BOT","message":"\xec\xa0\x80\xec\x99\x80 \xec\xa0\x9c \xec\xb9\x9c\xea\xb5\xac\xeb\x8a\x94 \xeb\xa7\xa4\xec\x9a\xb0 \xec\x9a\xb0\xec\x9a\xb8\xed\x95\x9c \xec\x83\x81\xed\x83\x9c\xec\x9e\x85\xeb\x8b\x88\xeb\x8b\xa4. \xec\x84\x9c\xeb\xa1\x9c \xec\x97\xb0\xeb\x9d\xbd\xed\x95\x98\xec\xa7\x80 \xec\x95\x8a\xea\xb3\xa0 \xec\xa7\x80\xeb\x82\xb4\xea\xb3\xa0 \xec\x9e\x88\xec\x96\xb4\xec\x84\x9c \xeb\x8b\xb5\xeb\x8b\xb5\xed\x95\x98\xea\xb3\xa0 \xec\x95\x88\xec\x93\xb0\xeb\x9f\xbd\xec\x8a\xb5\xeb\x8b\x88\xeb\x8b\xa4. \xea\xb7\xb8\xeb\xa6\xac\xea\xb3\xa0 \xec\x96\xbc\xeb\xa7\x88 \xec\xa0\x84\xea\xb9\x8c\xec\xa7\x80\xeb\xa7\x8c \xed\x95\xb4\xeb\x8f\x84 \xea\xb4\x9c\xec\xb0\xae\xec\x95\x98\xeb\x8a\x94\xeb\x8d\xb0 \xec\x9d\xb4\xec\x95\xbc\xea\xb8\xb0\xea\xb0\x80 \xec\x83\x9c\xeb\x8b\xa4\xea\xb0\x80 \xec\x9a\x94\xec\xa6\x98 \xeb\x8b\xa4\xec\x8b\x9c \xed\x99\x94\xed\x95\xb4 \xeb\xaa\xa8\xeb\x93\x9c\xeb\xa1\x9c \xeb\x8f\x8c\xec\x95\x84\xea\xb0\x94\xeb\x82\x98\xec\x9a\x94"}],"stream":true,"max_token":256,"temperature":0.8,"timestamp":"1690111922558","top_k":20}', headers=[('target_model', b'prod_a'), ('req_id', b'24367d12-8f3b-4dfd-b4d9-4a81cd02ad5e'), ('__TypeId__', b'io.teammistake.suzume.data.InferenceRequest')], checksum=None, serialized_key_size=-1, serialized_value_size=436, serialized_header_size=113)
# get req
        if time.time()*1000-msg.timestamp>int(timeout_ms):
            print("TIMEOUTED CONTENT DELIVERED")
            continue
        
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

        if stream == None:
            stream = parsed['stream']
            max_token = parsed["max_token"]
            temperature = parsed["temperature"]
            top_k: int = parsed["top_k"]
        elif stream != parsed["stream"]:
            print(stream, parsed["stream"])
            continue

        req_ids.append(req_id)        
        req = parsed['req']

        context = ""
        # temp_req = req

        # # try: 
        # #     temp_context = parsed["context"].reverse()

        # #     for _context in temp_context:
        # #         if _context["type"] != "HUMAN": continue
        # #         temp_req = f"{_context['message']} {temp_req}"

        # #         if len(temp_req) < 200:    
        # #             req = temp_req       
        # #         else: break
        # # except: pass
        # print(req)

        reqs.append(req)
        print("get req")
    # except TimeoutError:
    #     print("timeout error")
    #     # Handle the timeout exception
    #     if len(req_ids) == 0:
    #         continue
    # else: 
    #     print("WHY?")        
    #     continue

    if len(req_ids)==0:
        continue

    print("Generation Start")
    if stream:
        print("Streaming Creation")

        for req_id in req_ids:
            producer.send(RESP_TOPIC, json.dumps({
                'resp_partial': "",
                'resp_full': "",
                'eos': False
            }).encode(), headers=[("req_id", req_id.encode()), ("seq_id", '1'.encode())])

        seq = 1
        already = list(range(len(reqs)))
        done = [0] * len(already)
        total = {key:"" for key in already}
        for (parts, eoses) in model.generator(reqs, max_token=max_token, temperature=temperature, top_k=top_k):
            seq += 1
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
