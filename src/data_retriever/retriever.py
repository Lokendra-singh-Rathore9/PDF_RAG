import sys
import os
from operator import itemgetter
from typing import List, Optional, Dict, Any

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS

from prompt.prompt_library import PROMPT_REGISTRY
from logger import GLOBAL_LOGGER as log
from dotenv import load_dotenv
load_dotenv()