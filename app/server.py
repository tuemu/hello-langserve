#!/usr/bin/env python
from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langserve import add_routes
from fastapi.middleware.cors import CORSMiddleware
from app.role_based_cooperation.main import RoleBasedCooperation
from app.settings import Settings

settings = Settings()

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

llm = ChatOpenAI(
    model=settings.openai_mini_model, temperature=settings.temperature
)

agent = RoleBasedCooperation(llm=llm)

add_routes(
    app,
    agent.graph,
    path="/my-agent",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8080)