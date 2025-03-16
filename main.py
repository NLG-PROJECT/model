import os
import uuid
import pickle
from typing import List, Dict, Any
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from pydantic import BaseModel, HttpUrl
from PyPDF2 import PdfReader
from docx import Document  # Ensure using python-docx
import requests
from langchain_ollama import OllamaEmbeddings
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pathlib import Path
import json
import logging
import sys
from groq import Groq  # New import for Groq client
import subprocess
import time
import signal
from fastapi import Query
from urllib.parse import urljoin, urlparse
from collections import deque
import re
import redis
from redis.commands.search.field import VectorField, TextField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
app = FastAPI()
# Initialize logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for verbose output
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),  # Log to stdout
    ]
)