"""
performance_utils.py

Unified inference module using vLLM for both simple and thinking modes
พร้อมใช้งานบน RunPod Serverless รองรับ JSON input แบบ OpenAI Chat API และ
รองรับ task_type (blog, fb_post, ad) พร้อม token optimization และ
รองรับ override max_tokens และ override think_mode จาก client
"""
import os
import runpod
from vllm import LLM, SamplingParams
from typing import List, Dict, Any
from token_utils import compute_remaining_tokens

class InferenceHandler:
    """
    จัดการ inference ทั้ง simple และ thinking mode ด้วย vLLM:
    - โหลดโมเดลครั้งเดียว ลดการใช้ GPU memory ซ้ำซ้อน
    - รองรับ system prompt จาก input
    - ปรับ temperature และ budget ตาม request
    """
    def __init__(
        self,
        model_name: str,
        dtype: str = 'bfloat16',
        max_think_tokens: int = 2048,
        max_ignore: int = 5
    ):
        self.max_think_tokens = max_think_tokens
        self.max_ignore = max_ignore
        # โหลดโมเดล vLLM เพียงครั้งเดียว
        self.model = LLM(model_name, dtype=dtype, enforce_eager=True)

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        think: bool
    ) -> str:
        # แยก system prompt และ user content
        system_prompts = [m['content'] for m in messages if m['role']=='system']
        user_texts = [m['content'] for m in messages if m['role']!='system']
        system = system_prompts[0] if system_prompts else 'You are a helpful assistant.'
        prompt_body = ' '.join(user_texts)

        if think:
            # Thinking mode (chain-of-thought)
            tpl = self.model.tokenizer.apply_chat_template(
                [
                    {'role':'system','content':system},
                    {'role':'user','content':prompt_body}
                ],
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=True
            )
            # รอบคิด
            sam = SamplingParams(
                max_tokens=self.max_think_tokens,
                temperature=temperature,
                seed=0,
                stop=['</think>']
            )
            out = self.model.generate([tpl], sampling_params=sam)[0]
            tokens_used = len(out.outputs[0].token_ids)
            tpl += out.outputs[0].text
            # รอบ ignore (เติม Alternatively)
            for _ in range(self.max_ignore):
                left = self.max_think_tokens - tokens_used
                if left <= 0:
                    break
                tpl += '\nAlternatively'
                sam2 = SamplingParams(
                    max_tokens=left,
                    temperature=temperature,
                    seed=0,
                    stop=['</think>']
                )
                out2 = self.model.generate([tpl], sampling_params=sam2)[0]
                tokens_used += len(out2.outputs[0].token_ids)
                tpl += out2.outputs[0].text
            # รอบตอบ
            tpl += "\nTime's up. End of thinking process. Will answer immediately.\n</think>"
            sam3 = SamplingParams(max_tokens=max_tokens, temperature=temperature, seed=0)
            out3 = self.model.generate([tpl], sampling_params=sam3)[0]
            return tpl + out3.outputs[0].text
        else:
            # Simple mode
            tpl = self.model.tokenizer.apply_chat_template(
                [
                    {'role':'system','content':system},
                    {'role':'user','content':prompt_body}
                ],
                add_generation_prompt=True,
                tokenize=False
            )
            sam = SamplingParams(max_tokens=max_tokens, temperature=temperature, seed=0)
            out = self.model.generate([tpl], sampling_params=sam)[0]
            return tpl + out.outputs[0].text

# อ่าน configuration จาก environment
MODEL_NAME = os.getenv('MODEL_NAME','scb10x/typhoon2.1-gemma3-4b')
MAX_THINK = int(os.getenv('MAX_THINK_TOKEN','2048'))
MAX_IGNORE = int(os.getenv('MAX_IGNORE','5'))

# สร้าง InferenceHandler
inf_handler = InferenceHandler(
    MODEL_NAME,
    dtype=os.getenv('DTYPE','bfloat16'),
    max_think_tokens=MAX_THINK,
    max_ignore=MAX_IGNORE
)

# RunPod serverless handler

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    inp = job.get('input', {})
    messages = inp.get('messages')
    temperature = float(inp.get('temperature', 0.6))
    task_type = inp.get('task_type', '').lower()
    # รับ optional user-defined max_tokens
    user_max = inp.get('max_tokens')
    # รับ optional user-defined think flag
    user_think = inp.get('think')

    if not messages:
        return {'error':'`messages` required'}

    # คำนวณ remaining tokens
    remaining = compute_remaining_tokens(messages, context_limit=8192)
    if remaining <= 0:
        return {'error': 'Input exceeds context window limit.'}

    # กำหนด default max output tokens ตามประเภทงาน
    defaults = {'blog':800, 'fb_post':150, 'ad':60}
    default_max = defaults.get(task_type, remaining)

    # ถ้ามี user-defined max_tokens ให้ใช้ค่านั้น, มิฉะนั้นใช้ default
    if user_max is not None:
        max_tokens = int(user_max)
    else:
        max_tokens = min(default_max, int(remaining * 0.95))
    # ตัดไม่ให้เกิน remaining
    max_tokens = min(max_tokens, remaining)

    # กำหนด think_mode: ถ้า user ส่ง flag ให้ใช้ ค่านั้น; ไม่งั้น blog=True, อื่น=False
    if isinstance(user_think, bool):
        think_mode = user_think
    else:
        think_mode = (task_type == 'blog')

    try:
        result = inf_handler.generate(messages, temperature, max_tokens, think_mode)
        return {'generated_text': result}
    except Exception as e:
        return {'error': str(e)}

# เริ่ม RunPod serverless
runpod.serverless.start({'handler': handler})
