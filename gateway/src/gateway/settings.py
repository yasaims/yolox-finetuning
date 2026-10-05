from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    trainer_addr: str = "trainer:50051"
