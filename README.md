# Introduction
In this project a user can upload multiple PDF and can chat with its content.

![image](https://github.com/atiqbaqi/chat-with-pdf-streamlit/assets/73009994/12dc9aab-1743-44e4-bba8-158f12575ade)


## Project installation

Create a virtual environment
```
python -m venv .venv
```
Activate the virtual environment(for windows machine)
```
.\.venv\Scripts\activate
```

Then install dependencies

```
pip install -r requirements.txt
```
After that create `.env` file in your project for secret API keys for OpenAI or HuggingFace. Get your API keys from respective platform using your account. You have to follow the API key naming convention, for OpenAI its `OPENAI_API_KEY` and for HuggingFace `HUGGINGFACEHUB_API_TOKEN`.

The frontend of this project is done using `streamlit`. TO run the application use following command
```
streamlit run app.py
```

## How it works
![image](https://github.com/atiqbaqi/chat-with-pdf-streamlit/assets/73009994/d7e5475d-9adc-40de-af67-c1fadd173fdf)

