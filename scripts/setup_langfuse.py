"""Generate local credentials without printing or replacing secrets."""

from pathlib import Path
import secrets


def main():
    root = Path(__file__).resolve().parents[1]
    destination = root / "deploy/.env.langfuse"
    values = {
        key: secrets.token_hex(32)
        for key in (
            "POSTGRES_PASSWORD",
            "CLICKHOUSE_PASSWORD",
            "REDIS_AUTH",
            "MINIO_ROOT_PASSWORD",
            "SALT",
            "ENCRYPTION_KEY",
            "NEXTAUTH_SECRET",
            "LANGFUSE_INIT_USER_PASSWORD",
        )
    }
    values.update(
        {
            "NEXTAUTH_URL": "http://127.0.0.1:3000",
            "LANGFUSE_INIT_ORG_ID": "tracemind-local",
            "LANGFUSE_INIT_ORG_NAME": "TraceMind",
            "LANGFUSE_INIT_PROJECT_ID": "tracemind-evaluation",
            "LANGFUSE_INIT_PROJECT_NAME": "TraceMind Evaluation",
            "LANGFUSE_INIT_PROJECT_PUBLIC_KEY": "pk-lf-"
            + secrets.token_hex(16),
            "LANGFUSE_INIT_PROJECT_SECRET_KEY": "sk-lf-"
            + secrets.token_hex(32),
            "LANGFUSE_INIT_USER_EMAIL": "tracemind@localhost.invalid",
            "LANGFUSE_INIT_USER_NAME": "TraceMind Local",
        }
    )
    values.update(
        {
            "TRACEMIND_LLM_PROVIDER": "deepseek",
            "TRACEMIND_LLM_MODEL": "deepseek-v4-pro",
            "TRACEMIND_AGENT_FALLBACK_PROVIDER": "deepseek",
            "TRACEMIND_AGENT_FALLBACK_MODEL": "deepseek-flash",
            "TRACEMIND_EVAL_JUDGE_PROVIDER": "deepseek",
            "TRACEMIND_EVAL_JUDGE_MODEL": "deepseek-v4-pro",
        }
    )
    values.update(
        {
            "TRACEMIND_LANGFUSE_ENABLED": "true",
            "TRACEMIND_LANGFUSE_BASE_URL": "http://127.0.0.1:3000",
            "TRACEMIND_LANGFUSE_PUBLIC_KEY": values[
                "LANGFUSE_INIT_PROJECT_PUBLIC_KEY"
            ],
            "TRACEMIND_LANGFUSE_SECRET_KEY": values[
                "LANGFUSE_INIT_PROJECT_SECRET_KEY"
            ],
        }
    )
    try:
        with destination.open("x") as handle:
            destination.chmod(0o600)
            handle.write(
                "\n".join(f"{key}={value}" for key, value in values.items())
                + "\n"
            )
    except FileExistsError:
        raise SystemExit(
            "Existing deploy/.env.langfuse retained; no credentials changed"
        )
    print(
        "Created deploy/.env.langfuse (mode 0600). "
        "Credentials were not printed."
    )


if __name__ == "__main__":
    main()
