import math
from typing import List, Dict, Any, Sequence

import backoff
import requests
from openai import OpenAI

from agent.utils import *
from templates.android_screenshot_template import *


def _logsumexp(values: Sequence[float]) -> float:
    if not values:
        return float("-inf")
    peak = max(values)
    if peak == float("-inf"):
        return peak
    return peak + math.log(sum(math.exp(value - peak) for value in values))


def _softmax(scores: Sequence[float]) -> List[float]:
    total = _logsumexp(scores)
    if total == float("-inf"):
        return [1.0 / len(scores) for _ in scores] if scores else []
    return [math.exp(score - total) for score in scores]


def handle_giveup(details):
    print(
        "Backing off {wait:0.1f} seconds afters {tries} tries calling fzunction {target} with args {args} and kwargs {kwargs}"
        .format(**details))


def handle_backoff(details):
    args = str(details['args'])[:1000]
    print(f"Backing off {details['wait']:0.1f} seconds after {details['tries']} tries "
          f"calling function {details['target'].__name__} with args {args} and kwargs ")

    import traceback
    print(traceback.format_exc())


class Agent:
    name: str

    @backoff.on_exception(
        backoff.expo, Exception,
        on_backoff=handle_backoff,
        on_giveup=handle_giveup,
    )
    def act(self, messages: List[Dict[str, Any]]) -> str:
        raise NotImplementedError

    def prompt_to_message(self, prompt, images):
        raise NotImplementedError

    def system_prompt(self, instruction) -> str:
        raise NotImplementedError


class OpenAIAgent(Agent):
    def __init__(
            self,
            api_key: str = '',
            api_base: str = '',
            model_name: str = '',
            max_new_tokens: int = 16384,
            temperature: float = 0,
            top_p: float = 0.7,
            chat_template_kwargs: Dict[str, Any] | None = None,
            presence_penalty: float | None = None,
            frequency_penalty: float | None = None,
            repetition_penalty: float | None = None,
            top_k: int | None = None,
            min_p: float | None = None,
            top_logprobs: int = 20,
            request_timeout: float = 120,
            extra_body: Dict[str, Any] | None = None,
            **kwargs
    ) -> None:
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        # openai.api_base = api_base
        # openai.api_key = api_key
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.chat_template_kwargs = chat_template_kwargs
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.repetition_penalty = repetition_penalty
        self.top_k = top_k
        self.min_p = min_p
        self.top_logprobs = top_logprobs
        self.request_timeout = request_timeout
        self.extra_body = extra_body or {}
        self.kwargs = kwargs
        self.name = "OpenAIAgent"

    def _openai_body(self, messages: List[Dict[str, Any]], max_tokens: int | None = None,
                     temperature: float | None = None) -> Dict[str, Any]:
        body = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_new_tokens if max_tokens is None else max_tokens,
            "temperature": self.temperature if temperature is None else temperature,
            "top_p": self.top_p,
        }
        if self.presence_penalty is not None:
            body["presence_penalty"] = self.presence_penalty
        if self.frequency_penalty is not None:
            body["frequency_penalty"] = self.frequency_penalty

        extra_body = dict(self.extra_body)
        if self.chat_template_kwargs is not None:
            extra_body["chat_template_kwargs"] = self.chat_template_kwargs
        if self.repetition_penalty is not None:
            extra_body["repetition_penalty"] = self.repetition_penalty
        if self.top_k is not None:
            extra_body["top_k"] = self.top_k
        if self.min_p is not None:
            extra_body["min_p"] = self.min_p
        body.update(extra_body)
        return body

    def _chat_url(self) -> str:
        base = str(self.client.base_url).rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.client.api_key}",
            "Content-Type": "application/json",
        }

    def _post_chat(self, body: Dict[str, Any]) -> Dict[str, Any]:
        response = requests.post(
            self._chat_url(),
            headers=self._headers(),
            json=body,
            timeout=self.request_timeout,
        )
        response.raise_for_status()
        return response.json()

    def _extract_next_token_logits(self, data: Dict[str, Any]) -> Dict[str, float]:
        choices = data.get("choices") or []
        choice = choices[0] if choices else {}
        logits = choice.get("next_token_logits") or data.get("next_token_logits")
        if isinstance(logits, dict):
            out: Dict[str, float] = {}
            for key, value in logits.items():
                if isinstance(value, dict):
                    token = value.get("decoded_token") or value.get("token") or key
                    logprob = value.get("logprob")
                else:
                    token = key
                    logprob = value
                if logprob is not None:
                    out[str(token)] = float(logprob)
            return out

        prefill = choice.get("prefill_logits") or data.get("prefill_logits")
        if isinstance(prefill, list) and prefill:
            last = prefill[-1]
            if isinstance(last, dict):
                return {str(key): float(value) for key, value in last.items()}
            if isinstance(last, list):
                out: Dict[str, float] = {}
                for item in last:
                    if isinstance(item, dict):
                        token = item.get("token") or item.get("text")
                        value = item.get("logprob")
                        if token is not None and value is not None:
                            out[str(token)] = float(value)
                if out:
                    return out

        token_logprobs = ((choice.get("logprobs") or {}).get("content") or [])
        if token_logprobs:
            top = token_logprobs[0].get("top_logprobs") or []
            out = {}
            for item in top:
                token = item.get("token")
                value = item.get("logprob")
                if token is not None and value is not None:
                    out[str(token)] = float(value)
            return out
        raise RuntimeError(f"Response does not contain next-token logits: {str(data)[:500]}")

    def score_letters(self, messages: List[Dict[str, Any]], letters: Sequence[str]) -> List[Dict[str, float]]:
        body = self._openai_body(messages, max_tokens=1, temperature=0)
        body.update({
            "stream": False,
            "logprobs": True,
            "top_logprobs": self.top_logprobs,
            "vllm_xargs": {"return_prefill_logits": 1},
        })
        data = self._post_chat(body)
        logits = self._extract_next_token_logits(data)
        scores: List[float] = []
        for letter in letters:
            scores.append(logits.get(str(letter), float("-inf")))
        probs = _softmax(scores)
        return [
            {"letter": str(letter), "logit": float(score), "prob": float(prob)}
            for letter, score, prob in zip(letters, scores, probs)
        ]

    def score_yes_no(self, messages: List[Dict[str, Any]]) -> Dict[str, float]:
        body = self._openai_body(messages, max_tokens=1, temperature=0)
        body.update({
            "stream": False,
            "logprobs": True,
            "top_logprobs": self.top_logprobs,
            "vllm_xargs": {"return_prefill_logits": 1},
        })
        data = self._post_chat(body)
        logits = self._extract_next_token_logits(data)
        yes_logit = logits.get("Yes", float("-inf"))
        no_logit = logits.get("No", float("-inf"))
        yes_prob, no_prob = _softmax([yes_logit, no_logit])
        return {
            "yes_logit": float(yes_logit),
            "no_logit": float(no_logit),
            "yes_prob": float(yes_prob),
            "no_prob": float(no_prob),
        }

    def generate_text(self, messages: List[Dict[str, Any]], max_tokens: int = 128,
                      temperature: float | None = None) -> str:
        body = self._openai_body(messages, max_tokens=max_tokens, temperature=temperature)
        data = self._post_chat(body)
        choices = data.get("choices") or []
        if not choices:
            return ""
        return ((choices[0].get("message") or {}).get("content") or "").strip()


    @backoff.on_exception(
        backoff.expo, Exception,
        on_backoff=handle_backoff,
        on_giveup=handle_giveup,
        max_tries=10
    )
    def act(self, messages: List[Dict[str, Any]]) -> str:
        request = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if self.presence_penalty is not None:
            request["presence_penalty"] = self.presence_penalty
        if self.frequency_penalty is not None:
            request["frequency_penalty"] = self.frequency_penalty

        extra_body = dict(self.extra_body)
        if self.chat_template_kwargs is not None:
            extra_body["chat_template_kwargs"] = self.chat_template_kwargs
        if self.repetition_penalty is not None:
            extra_body["repetition_penalty"] = self.repetition_penalty
        if self.top_k is not None:
            extra_body["top_k"] = self.top_k
        if self.min_p is not None:
            extra_body["min_p"] = self.min_p
        if extra_body:
            request["extra_body"] = extra_body

        r = self.client.chat.completions.create(**request)
        return r.choices[0].message.content

    def prompt_to_message(self, prompt, images):
        content = [
            {
                "type": "text",
                "text": prompt
            }
        ]
        for img in images:
            base64_img = image_to_base64(img)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_img}"
                }
            })
        message = {
            "role": "user",
            "content": content
        }
        return message

    def system_prompt(self, instruction) -> str:
        return SYSTEM_PROMPT_ANDROID_MLLM_DIRECT + f"\n\nTask Instruction: {instruction}"


class HTTPAgent(Agent):
    def __init__(
            self,
            url: str,
            headers: Dict[str, Any] = {},
            body: Dict[str, Any] = {},
            return_format: str = "{response[choices][0][message][content]}"
    ) -> None:
        self.url = url
        self.headers = headers
        self.body = body
        self.return_format = return_format

    def update_messages(self, body: Dict[str, Any], messages: List[Dict[str, Any]]):
        body.update({"messages": messages})
        return body

    @backoff.on_exception(
        backoff.expo, Exception,
        on_backoff=handle_backoff,
        on_giveup=handle_giveup,
    )
    def act(self, messages: List[Dict[str, Any]]):
        body = self.update_messages({**self.body}, messages)
        response = requests.post(
            self.url, headers=self.headers, body=body
        )
        return self.return_format.format(response=response)
