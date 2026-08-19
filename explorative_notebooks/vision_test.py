import argparse
import base64
import json
import os
import urllib.request


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def call_openrouter_vision(
    image_paths,
    prompt,
    model,
):
    api_key = os.environ.get("LLM_ROUTER_API_KEY")

    if not api_key:
        raise RuntimeError(
            "LLM_ROUTER_API_KEY not set"
        )

    content = [
        {
            "type": "text",
            "text": prompt,
        }
    ]

    for image_path in image_paths:

        with open(image_path, "rb") as f:
            base64_image = base64.b64encode(
                f.read()
            ).decode("utf-8")

        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": (
                        f"data:image/png;base64,"
                        f"{base64_image}"
                    )
                },
            }
        )

    payload = {
        "model": model,
        "max_tokens": 1000,
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
    }

    req = urllib.request.Request(
        url="https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "stage3-vision-test",
            "X-Title": "vision-test",
        },
        method="POST",
    )

    try:

        with urllib.request.urlopen(
            req,
            timeout=120,
        ) as resp:

            response = json.loads(
                resp.read().decode("utf-8")
            )

            return (
                response["choices"][0]
                ["message"]
                ["content"]
            )

    except urllib.error.HTTPError as exc:

        body = exc.read().decode(
            "utf-8",
            errors="replace",
        )

        print(body)
        raise


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "images",
        nargs="+"
    )

    parser.add_argument(
        "--model",
        default="anthropic/claude-sonnet-4.6",
    )

    parser.add_argument(
        "--prompt",
        default="""
    Extract all cost-related datapoints from this figure.

    Return ONLY valid JSON.

    {
      "costs": [
        {
          "Cost_Type": "",
          "Scenario_Name": "",
          "Value": "",
          "Value_Unit": "",
          "Year": "",
          "Area": ""
        }
      ]
    }
    """,
    )

    args = parser.parse_args()

    result = call_openrouter_vision(
        image_paths=args.images,
        prompt=args.prompt,
        model=args.model,
    )

    print()
    print("=" * 80)
    print(result)
    print("=" * 80)


if __name__ == "__main__":
    main()