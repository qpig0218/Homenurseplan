"""Flask application exposing an Azure OpenAI powered scheduling endpoint."""
import os
from functools import lru_cache
from typing import Any, Dict, List

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from openai import AzureOpenAI

load_dotenv()

app = Flask(__name__)

SYSTEM_PROMPT = (
    "角色\n"
    "你是一位專業的路線排程專家，可以洞察醫療端以及民眾段的需求以及人性，並熟知交通不同時段狀況以及當地文化習慣，"
    "總是用盡全力與相關背景知識與工具來逐步推理出令人滿意的訪視路線。\n\n\n"
    "格式\n"
    "1. 以臺灣繁體中文來呈現“訪視時間表”、“訪視路線排程結果”以及“排程考量與貼心提醒”。\n"
    "2. 訪視時間表必須包含這些項目：順序、地點類別、名稱/病人姓名、地址、時間。\n"
    "3. 時間格式為24小時制，HH：MM 。\n"
    "4. ”訪視路線排程結果“則是會在訪視時間表下方呈現一個連結按鈕（排程路線地圖），點按之後可以連結到Google地圖應用程式或是網頁，呈現訪視路線地圖。\n"
    "5. “排程考量與貼心提醒”：簡易說明這份訪視路線排程要注意的地方。\n\n\n\n"
    "指令\n"
    "1. 必須遵守角色設定，還有以臺灣繁體中文來呈現方式時間表以及路線排程結果。\n"
    "2.先置放完優先事項的時間點或是特殊訪視時段要求以及限制之後才能開始排程。\n"
    "3. 排程前必須先調用Google Map的Function去得到任兩點的交通距離以及時間。\n"
    "4. 依據特殊個案的優先指定時段還有得到的任兩點交通距離時間再來遵守「拓撲排序」邏輯來編排訪視路線。\n"
    "5.訪視後的路線必須完成特殊個案的優先指定時段以及最佳經濟效率的路線排班。\n"
    "6. 排完路線後，需要呈現完整詳細的訪視時間表以及在Google 地圖上畫出訪視路線地圖。\n"
    "7. 訪視時間表必須包含這些項目：順序、地點類別、名稱/病人姓名、地址、時間。\n"
    "8. 順序：保留項目名稱為阿拉伯數字，出發點為0。\n"
    "9. 地點類別：保留項目名稱，項目包含：出發點、病人、用餐、終點。\n"
    "10.名稱/病人姓名：保留項目名稱，項目包含：起點名稱、病人姓名（+特殊時段需求/極簡背景備註）、終點名稱\n"
    "11. 地址：保留項目名稱，寫出對應的地址資訊，用餐地址空白即可。\n"
    "12. 時間：保留項目名稱，項目包含起點出發時間（出發點）、抵達時間（病人）、訪視停留時間（病人）、離開時間（彬病人）、用餐時間（移動緩衝時間）、抵達時間（終點）。時間格式為24小時制，HH：MM 。\n"
    "13. ”訪視路線排程結果“則是會在訪視時間表下方呈現一個連結按鈕（排程路線地圖），點按之後可以連結到Google地圖應用程式或是網頁，呈現訪視路線地圖。\n"
    "14. “排程考量與貼心提醒”：在”訪視路線排程結果“之後，簡易說明這份訪視路線排程要注意的地方，禁止說到拓撲兩個字，只需要說名是否有符合特殊時段需求以及最佳路線排程即可。\n"
    "15. 病人訪視一定要嚴格遵守「拓撲排序」邏輯。"
)


class ConfigurationError(RuntimeError):
    """Raised when mandatory Azure OpenAI configuration is missing."""


@lru_cache(maxsize=1)
def build_client() -> Dict[str, Any]:
    """Create a cached Azure OpenAI client and return it with deployment info."""
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("ENDPOINT_URL")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT") or os.getenv("DEPLOYMENT_NAME")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")

    if not endpoint or not deployment or not api_key:
        raise ConfigurationError(
            "Azure OpenAI 設定不完整，請確認 AZURE_OPENAI_ENDPOINT、AZURE_OPENAI_DEPLOYMENT 以及 AZURE_OPENAI_API_KEY 已設定於環境變數或 .env。"
        )

    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )
    return {"client": client, "deployment": deployment}


def _normalize_messages(user_messages: List[Dict[str, Any]], prompt: str) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                }
            ],
        }
    ]

    for message in user_messages:
        role = message.get("role")
        content = message.get("content")
        if not role or content is None:
            continue
        if isinstance(content, str):
            content_payload = [{"type": "text", "text": content}]
        elif isinstance(content, list):
            content_payload = content
        else:
            continue
        messages.append({"role": role, "content": content_payload})

    if prompt:
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ],
            }
        )

    return messages


@app.post("/api/route-plan")
def generate_route_plan():
    payload = request.get_json(silent=True) or {}
    prompt = (payload.get("prompt") or payload.get("user_prompt") or "").strip()
    user_messages = payload.get("messages") or []

    if not prompt and not user_messages:
        return (
            jsonify({"error": "請提供至少一段訪視需求說明或歷史對話。"}),
            400,
        )

    messages = _normalize_messages(user_messages if isinstance(user_messages, list) else [], prompt)

    try:
        config = build_client()
        completion = config["client"].chat.completions.create(
            model=config["deployment"],
            messages=messages,
            max_completion_tokens=payload.get("max_completion_tokens", 4096),
            temperature=payload.get("temperature", 0.2),
        )
    except ConfigurationError as config_error:
        return jsonify({"error": str(config_error)}), 500
    except Exception as exc:  # pragma: no cover - network errors
        return (
            jsonify({"error": "Azure OpenAI 服務呼叫失敗", "detail": str(exc)}),
            502,
        )

    choice = completion.choices[0] if completion.choices else None
    message_content = ""
    if choice and choice.message:
        message_content = choice.message.content or ""

    response_payload: Dict[str, Any] = {
        "content": message_content,
        "usage": getattr(completion, "usage", None),
    }

    if choice and getattr(choice, "finish_reason", None):
        response_payload["finish_reason"] = choice.finish_reason

    if choice and getattr(choice, "message", None):
        response_payload["message_role"] = choice.message.role

    return jsonify(response_payload)


@app.get("/health")
def health_check():
    return jsonify({"status": "ok"})


if __name__ == "__main__":  # pragma: no cover
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
