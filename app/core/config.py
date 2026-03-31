from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    ollama_base_url: str = "http://localhost:11434"
    ollama_vision_model: str = "llava:7b"
    ollama_text_model: str = "llama3.2:latest"
    ollama_embed_model: str = "nomic-embed-text"

    lancedb_path: str = "./lancedb"
    lancedb_table_name: str = "turaco_policies"

    sqlite_db_path: str = "./claims.db"

    app_env: str = "development"
    log_level: str = "INFO"

    confidence_threshold: float = 0.75

    model_config = {"env_file": ".env"}


settings = Settings()
