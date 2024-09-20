import os
import sys

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
project_root = os.path.dirname(current_dir)
# 将 src 目录添加到 Python 路径
sys.path.append(os.path.join(project_root, 'src'))


from fastapi import FastAPI, Form, Depends, HTTPException, Request, status, File, UploadFile, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, constr
from typing import Optional
from passlib.context import CryptContext
from motor.motor_asyncio import AsyncIOMotorClient
from bson.objectid import ObjectId
from jose import JWTError, jwt
from datetime import datetime, timedelta
import logging
import os, sys
from handwriting_recognition import HandwritingRecognizer, data_cleaning
import tempfile
import base64
from PIL import Image
import io, shutil,re
from typing import List
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



current_dir = os.path.dirname(os.path.abspath(__file__))
# 挂载静态文件目录
app.mount("/static", StaticFiles(directory=os.path.join(current_dir, "static")), name="static")
# Jinja2模板
templates = Jinja2Templates(directory=os.path.join(current_dir, "templates"))

# MongoDB连接
MONGO_DETAILS = "mongodb://localhost:27017"
client = AsyncIOMotorClient(MONGO_DETAILS)
database = client.users_db
users_collection = database.get_collection("users_collection")

logging.basicConfig(filename='log.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LogMessage(BaseModel):
    message: str

@app.post("/log")
async def log_message(log: LogMessage):
    logger.info(f"Frontend log: {log.message}")
    return {"message": "Log received"}


TEMP_DIR = os.path.join(os.path.dirname(__file__), 'temp')
# 确保临时目录存在
os.makedirs(TEMP_DIR, exist_ok=True)
recognizer = HandwritingRecognizer()

database = client.ai_training
handwriting_training_data_collection = database.get_collection("handwriting_training_data")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("handwriting.html", {"request": request})

# ------------------------
class RecognitionResult(BaseModel):
    line_num: int
    text: str
    status: str
    image_path: Optional[str] = None

@app.get("/handwriting")
async def handwriting_page(request: Request):
    return templates.TemplateResponse("handwriting.html", {"request": request})


@app.post("/handwriting")
async def recognize_handwriting(file: UploadFile = File(...), kernel_width: int = Form(...)):
    # 创建一个临时目录，但不使用 with 语句
    temp_dir = tempfile.mkdtemp(dir=TEMP_DIR)
    
    try:
        # 保存上传的文件
        file_path = os.path.join(temp_dir, file.filename)
        with open(file_path, "wb") as buffer:
            contents = await file.read()
            buffer.write(contents)

        # 获取文件所在目录（即与file_path同级的目录）
        file_directory = os.path.dirname(file_path)

        # 调用识别函数，使用传入的 kernel_width
        results = recognizer.recognize_image(file_path, kernel_width=kernel_width)
        
        # 清理数据
        for key, re_txt in results.items():
            results[key] = data_cleaning(re_txt)
        
        # 读取生成的 visualized_text_regions.jpg
        pattern = r"(.+)\.jpg$"
        file_directory_temp = re.search(pattern, file_path, re.IGNORECASE).group(1) + '_temp'

        visualized_image_path = os.path.join(file_directory_temp, 'visualized_text_regions.jpg')
        if os.path.exists(visualized_image_path):
            with open(visualized_image_path, "rb") as image_file:
                visualized_encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        else:
            visualized_encoded_image = ""
            print(f"Warning: Image not found at {visualized_image_path}")
        
        # 获取 file_directory 中的所有 line_num.jpg 文件
        line_images = []
        pattern = r'line_(\d+)\.jpg$'
        for filename in os.listdir(file_directory_temp):
            match = re.match(pattern, filename)
            if match:
                line_num = int(match.group(1))
                image_path = os.path.join(file_directory_temp, filename)
                with open(image_path, "rb") as image_file:
                    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
                line_images.append({
                    'line_num': line_num,
                    'image': encoded_image
                })
        
        # 按行号排序
        line_images.sort(key=lambda x: x['line_num'], reverse= True)

        recognizer_result_json = {
            'recognizer_result': results,
            'visualized_image': visualized_encoded_image,
            'file_directory': file_directory_temp,
            'line_images': line_images
        }
        
        return JSONResponse(content=recognizer_result_json)
    except Exception as e:
        # 如果发生错误，删除临时目录
        shutil.rmtree(file_directory_temp, ignore_errors=True)
        raise e
        # os.unlink(temp_file_path)

@app.post("/submit_results")
async def submit_results(results: List[RecognitionResult] = Body(...)):
    try:
        # 处理结果
        processed_results = []
        training_data = []
        for result in results:
            processed_result = {
                "line_num": result.line_num,
                "text": result.text,
                "status": result.status,
                # "image_path": result.image_path,
            }
            # 如果提供了 image_path，则添加到结果中
            if result.image_path:
                processed_result["image_path"] = result.image_path

                training_item = {
                    "image_path": result.image_path,
                    "text": result.text,
                    "timestamp": datetime.utcnow()
                }
                training_data.append(training_item)

            processed_results.append(processed_result)
        
        # 按行号排序结果
        processed_results.sort(key=lambda x: x['line_num'])
        
        # 将训练数据存储到 MongoDB
        if training_data:
            # await handwriting_training_data_collection.insert_many(training_data)
            logger.info(f"Stored {len(training_data)} items in handwriting training data collection")
        
        # 记录日志
        logger.info(f"Processed results: {json.dumps(processed_results)}")

        return JSONResponse(content={
            "message": "Results processed successfully",
            "results": processed_results
        })
    except Exception as e:
        logger.error(f"Error processing results: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while processing results")



#------------------------------------------------------------------------


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)