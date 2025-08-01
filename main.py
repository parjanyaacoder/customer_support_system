import uvicorn
from fastapi import FastAPI, Request, Form 
from fastapi.responses import HTMLResponse
from fastapi.tempating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware 
from dotenv import load_dotenv

