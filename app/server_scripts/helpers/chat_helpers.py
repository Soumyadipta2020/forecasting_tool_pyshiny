# Helper functions for chat features
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import json
import time

def chat(user_message, history=None, system_prompt="general", api_key=None, temp=0.7):
    """
    Chat with OpenAI GPT model
    
    Args:
        user_message (str): User message
        history (list): Chat history
        system_prompt (str): System prompt type
        api_key (str): OpenAI API key
        temp (float): Temperature parameter
        
    Returns:
        str: Model response
    """
    if api_key is None:
        return "API key is required"
    
    # Get system prompt
    system = get_system_prompt(system_prompt)
    
    # Prepare prompt
    prompt = prepare_prompt(user_message, system, history)
    
    # API request
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    data = {
        "model": "gpt-3.5-turbo",
        "messages": prompt,
        "temperature": temp
    }
    
    # Make request with retry logic
    for attempt in range(4):
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            
            # Rate limit handling
            if response.status_code == 429:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
                
            return f"Error: {response.status_code} - {response.text}"
            
        except Exception as e:
            if attempt < 3:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                return f"Error: {str(e)}"

def chat_nvidia(user_message, history=None, api_key=None, model_llm="llama3-70b-instruct", 
                temp=0.2, topp=0.7, max_token=1024):
    """
    Chat with NVIDIA AI model
    
    Args:
        user_message (str): User message
        history (list): Chat history
        api_key (str): NVIDIA API key
        model_llm (str): Model name
        temp (float): Temperature parameter
        topp (float): Top-p parameter
        max_token (int): Maximum tokens
        
    Returns:
        str: Model response
    """
    if api_key is None:
        return "API key is required"
    
    # Prepare prompt
    user_prompt = [{"role": "user", "content": user_message}]
    prompt = (history or []) + user_prompt
    
    # API request
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    data = {
        "model": model_llm,
        "messages": prompt,
        "temperature": temp,
        "top_p": topp,
        "max_tokens": max_token
    }
    
    # Make request with retry logic
    for attempt in range(4):
        try:
            response = requests.post(
                "https://integrate.api.nvidia.com/v1/chat/completions",
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            
            # Rate limit handling
            if response.status_code == 429:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
                
            return f"Error: {response.status_code} - {response.text}"
            
        except Exception as e:
            if attempt < 3:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                return f"Error: {str(e)}"

def gemini(prompt, temperature=0.7, api_key=None, model="gemini-pro", max_retries=3):
    """
    Chat with Google Gemini model
    
    Args:
        prompt (str): User prompt
        temperature (float): Temperature parameter
        api_key (str): Google API key
        model (str): Model name
        max_retries (int): Maximum retries
        
    Returns:
        str: Model response
    """
    if api_key is None:
        return "API key is required"
    
    model_query = f"{model}:generateContent"
    
    # Global chat history
    global chat_history
    if not 'chat_history' in globals():
        chat_history = [{"role": "user", "parts": [{"text": prompt}]}]
    else:
        chat_history.append({"role": "user", "parts": [{"text": prompt}]})
    
    # Make request with retry logic
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model_query}",
                params={"key": api_key},
                headers={"Content-Type": "application/json"},
                json={
                    "contents": chat_history,
                    "generationConfig": {"temperature": temperature}
                }
            )
            
            if response.status_code == 200:
                answer = response.json()["candidates"][0]["content"]["parts"][0]["text"]
                chat_history.append({"role": "model", "parts": [{"text": answer}]})
                return answer
            
            # Retry with backoff
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                return f"Failed to access Gemini API after {max_retries} retries."
                
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                return f"Error: {str(e)}"

def get_system_prompt(system="general"):
    """
    Get system prompt based on type
    
    Args:
        system (str): System prompt type
        
    Returns:
        list: System prompt message
    """
    if system not in ["general", "code"]:
        system = "general"
        
    instructions = {
        "general": "You are a helpful assistant.",
        "code": "You are a helpful chat bot that answers questions for a Python programmer working in a Python IDE."
    }
    
    return [{"role": "system", "content": instructions[system]}]

def prepare_prompt(user_message, system_prompt, history):
    """
    Prepare prompt for chat models
    
    Args:
        user_message (str): User message
        system_prompt (list): System prompt
        history (list): Chat history
        
    Returns:
        list: Prepared prompt
    """
    user_prompt = [{"role": "user", "content": user_message}]
    return system_prompt + (history or []) + user_prompt

def update_history(history, user_message, response):
    """
    Update chat history
    
    Args:
        history (list): Current chat history
        user_message (str): User message
        response (str): Model response
        
    Returns:
        list: Updated chat history
    """
    new_history = (history or []) + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": response}
    ]
    
    return new_history 