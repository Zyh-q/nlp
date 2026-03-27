import base64
import os
from openai import OpenAI

client = OpenAI(
    base_url='https://api-inference.modelscope.cn/v1',
    api_key='ms-0d16614e-a7f3-4f1e-ae69-e566dba6cbbd', 
)

MODEL_ID = 'Qwen/Qwen3.5-27B'

def encode_image_to_base64(image_path):
    if not os.path.exists(image_path):
        return None
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def run_eco_agent(user_prompt, local_image_path=None):
    print(f"\n[EcoAgent] 正在感知输入并规划任务...")
    
    mcp_context = {
        "steps_today": 8500,
        "current_weather": "小雨",
        "user_emotion": "tired"
    }

    content = [{"type": "text", "text": user_prompt}]
    
    if local_image_path:
        base64_str = encode_image_to_base64(local_image_path)
        if base64_str:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_str}"}
            })

    system_instruction = f"""
    你是 EcoAgent，一个具备情感对齐能力的个人低碳助手。
    当前背景（来自MCP协议）：今日步数 {mcp_context['steps_today']}，天气 {mcp_context['current_weather']}。
    你的任务：
    1. 识别图片内容（如发票、食物等），分析其碳排放。
    2. 结合 MCP 背景数据。
    3. 如果用户表现出疲惫（当前状态：{mcp_context['user_emotion']}），请在回复中优先给予情感关怀，再给减碳建议。
    4. 所有的回复要像朋友一样亲切。
    """

    print(f"[EcoAgent] 正在调用 Qwen 大脑进行推理...\n")
    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": content}
        ],
        stream=True
    )

    print("[Agent 最终输出]:")
    print("-" * 30)
    for chunk in response:
        if chunk.choices:
            content = chunk.choices[0].delta.content
            if content:
                print(content, end='', flush=True)
    print("\n" + "-" * 30)

if __name__ == "__main__":
    test_image = "./lunch.jpg"
    test_text = "帮我看看这顿饭的碳足迹。"
    run_eco_agent(test_text, test_image if os.path.exists(test_image) else None)

