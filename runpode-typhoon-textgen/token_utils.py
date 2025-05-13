from token_utils import compute_remaining_tokens

def handler(job):
    inp = job.get("input", {})
    messages = inp.get("messages", [])
    temperature = float(inp.get("temperature", 0.6))

    # 1) ตรวจสอบ remaining tokens
    remaining = compute_remaining_tokens(messages, context_limit=8192)
    if remaining <= 0:
        return {"error": "Input exceeds context window limit."}

    # 2) กำหนด max_tokens จาก remaining_tokens (อาจเหลือ margin สักเล็กน้อย เช่น 5% เผื่อ stop tokens)
    max_tokens = int(remaining * 0.95)

    # 3) เลือกโหมด simple/think ตาม system prompt
    think_mode = any(m["role"] == "system" for m in messages)

    try:
        # เรียก inference ด้วย max_tokens ที่คำนวณได้
        result = inf_handler.generate(messages, temperature, max_tokens, think_mode)
        return {"generated_text": result}
    except Exception as e:
        return {"error": str(e)}
