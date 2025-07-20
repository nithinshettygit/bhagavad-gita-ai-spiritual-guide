# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Bhagavad Gita AI Backend",
    description="API for Chapter-wise Gita, Ask with Krishna AI, and dynamic content.",
    version="0.1.0"
)

# CORS configuration - Crucial for frontend-backend communication
origins = [
    "http://localhost:3000", # React/Vue development server
    # Add your Netlify frontend URL here when deployed, e.g., "https://your-frontend-app.netlify.app"
    "http://127.0.0.1:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Bhagavad Gita AI Backend!"}

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)