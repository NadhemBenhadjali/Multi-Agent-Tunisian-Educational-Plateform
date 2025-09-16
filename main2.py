import os
import nest_asyncio, uvicorn
from pyngrok import ngrok, conf
from fastapi import FastAPI
from app import app

if __name__ == "__main__":
    nest_asyncio.apply()
    os.environ['ngrok_authToken']='2yMaZ6btidIIiv3fwpkG287hAOT_2ezDgPqKcpGa2w9Z3WpxT'
    conf.get_default().ngrok_path = r"C:\ngrok\ngrok.exe"
    conf.get_default().auth_token = os.environ["ngrok_authToken"]
    public_url = ngrok.connect(8000)
    print("Public URL:", public_url)
    uvicorn.run(app, host="0.0.0.0", port=8000)
