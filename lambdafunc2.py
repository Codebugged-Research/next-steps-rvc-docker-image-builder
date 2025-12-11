import streamlit as st
import requests
import uuid
import time
import json

API_GATEWAY_URL = "Enter your api gateway url"
DEFAULT_PASSWORD = "next"

def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    
    if st.session_state.authenticated:
        return True
    
    st.title("Voice Conversion Dashboard")
    st.subheader("Login Required")
    
    password = st.text_input("Enter Password", type="password", key="password_input")
    
    if st.button("Login"):
        if password == DEFAULT_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password")
    
    return False

def start_ec2_and_get_endpoint():
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            response = requests.post(API_GATEWAY_URL, json={"action": "start"}, timeout=30)
            
            if response.status_code != 200:
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                return None
            
            data = response.json()
            body = json.loads(data.get('body', '{}'))
            status = body.get('status')
            endpoint = body.get('endpoint')
            
            if status == 'already_running':
                try:
                    health_check = requests.get(f"{endpoint}/health", timeout=5)
                    if health_check.status_code == 200:
                        st.success("Service is ready!")
                        return endpoint
                except:
                    st.info("Service initializing...")
                    time.sleep(15)
                    try:
                        health_check = requests.get(f"{endpoint}/health", timeout=5)
                        if health_check.status_code == 200:
                            st.success("Service is ready!")
                            return endpoint
                    except:
                        pass
                
                st.success("Service is ready!")
                return endpoint
            
            if status == 'starting':
                st.info("Starting service... Please wait 3-4 minutes")
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                max_wait_cycles = 48
                
                for i in range(max_wait_cycles):
                    time.sleep(10)
                    progress = min((i + 1) / max_wait_cycles, 0.95)
                    progress_bar.progress(progress)
                    
                    try:
                        status_response = requests.post(
                            API_GATEWAY_URL, 
                            json={"action": "status"}, 
                            timeout=30
                        )
                        
                        if status_response.status_code == 200:
                            status_data = json.loads(status_response.json().get('body', '{}'))
                            current_status = status_data.get('status')
                            
                            status_text.text(f"Status: {current_status} | Waited: {(i+1)*10}s")
                            
                            if current_status == 'running':
                                status_text.text("Service running, checking availability...")
                                time.sleep(5)
                                
                                for health_attempt in range(6):
                                    try:
                                        health_check = requests.get(f"{endpoint}/health", timeout=10)
                                        if health_check.status_code == 200:
                                            progress_bar.progress(1.0)
                                            status_text.empty()
                                            progress_bar.empty()
                                            st.success("Service is ready!")
                                            return endpoint
                                    except:
                                        status_text.text(f"Initializing... attempt {health_attempt + 1}/6")
                                        time.sleep(10)
                                
                                progress_bar.empty()
                                status_text.empty()
                                return endpoint
                    
                    except:
                        status_text.text(f"Checking... {(i+1)*10}s")
                        continue
                
                progress_bar.empty()
                status_text.empty()
                return None
            
            return endpoint
            
        except:
            if attempt < max_retries - 1:
                time.sleep(10)
                continue
            return None
    
    return None

def download_video_from_url(url):
    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    chunks = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            chunks.append(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                progress = downloaded / total_size
                progress_bar.progress(progress)
                status_text.text(f"Downloaded: {downloaded/1024/1024:.1f} MB / {total_size/1024/1024:.1f} MB")
    
    progress_bar.empty()
    status_text.empty()
    return b''.join(chunks)

def process_video_on_ec2(endpoint, video_bytes, gender, filename):
    files = {'video': (filename, video_bytes, 'video/mp4')}
    data = {'gender': gender}
    
    response = requests.post(
        f"{endpoint}/process",
        files=files,
        data=data,
        timeout=600
    )
    
    if response.status_code == 200:
        return response.content
    else:
        raise Exception(f"Processing failed")

def main():
    st.set_page_config(page_title="Voice Conversion", layout="centered")
    
    if not check_password():
        return
    
    st.title("Voice Conversion Dashboard")
    
    if st.button("Logout", key="logout_btn"):
        st.session_state.authenticated = False
        st.rerun()
    
    input_method = st.radio("Select Input Method", ["Upload Video File", "Video URL"])
    
    video_bytes = None
    filename = None
    
    if input_method == "Upload Video File":
        uploaded_file = st.file_uploader("Upload Video", type=['mp4', 'mov', 'avi', 'mkv'])
        if uploaded_file:
            video_bytes = uploaded_file.read()
            filename = uploaded_file.name
    else:
        video_url = st.text_input("Enter Video URL")
        if video_url:
            filename = f"video_{uuid.uuid4().hex[:8]}.mp4"
    
    voice_gender = st.selectbox("Voice Gender", ["Male", "Female"])
    
    if st.button("Process Video"):
        if input_method == "Upload Video File" and not video_bytes:
            st.warning("Please upload a video file")
            return
        
        if input_method == "Video URL" and not video_url:
            st.warning("Please enter a valid video URL")
            return
        
        try:
            if input_method == "Video URL":
                with st.spinner("Downloading video from URL..."):
                    video_bytes = download_video_from_url(video_url)
                    st.success(f"Downloaded {len(video_bytes)/1024/1024:.1f} MB")
            
            endpoint = start_ec2_and_get_endpoint()
            
            if not endpoint:
                st.warning("Unable to start service. Please try again.")
                return
            
            with st.spinner("Processing video (3-10 minutes)..."):
                output_video = process_video_on_ec2(endpoint, video_bytes, voice_gender, filename)
            
            st.success("Processing complete!")
            
            st.download_button(
                "Download Converted Video",
                output_video,
                f"converted_{filename}",
                "video/mp4"
            )
            
        except Exception as e:
            st.warning("Processing encountered an issue. Please try again.")
    
    st.divider()
    st.subheader("Service Control")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Check Service Status"):
            try:
                response = requests.post(API_GATEWAY_URL, json={"action": "status"}, timeout=30)
                if response.status_code == 200:
                    data = json.loads(response.json().get('body', '{}'))
                    status = data.get('status')
                    
                    if status == 'running':
                        st.success("Service Status: Active")
                    elif status == 'stopped':
                        st.info("Service Status: Inactive")
                    else:
                        st.warning(f"Service Status: {status}")
            except:
                st.warning("Unable to check status")
    
    with col2:
        if st.button("Stop Service"):
            try:
                response = requests.post(API_GATEWAY_URL, json={"action": "stop"}, timeout=30)
                if response.status_code == 200:
                    st.success("Service is stopping...")
            except:
                st.warning("Unable to stop service")

if __name__ == "__main__":
    main()
