import streamlit as st
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from google.generativeai import upload_file, get_file
import google.generativeai as genai
import time
from pathlib import Path
import tempfile
from dotenv import load_dotenv
load_dotenv()
import os

API_KEY=os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)

#PAGE CONFIG

st.set_page_config(
    page_title="Bhagya's video summariser",
    layout="wide"
)

st.title("PHIDATA Video Summarizer AI Agent")
st.header("Powered by Gemini 2.0 Flash Exp")

@st.cache_resource
def initializer_agent():
    return Agent(
        name="Video AI Summarizer",
        model=Gemini(id="gemini-2.0-flash-exp"),
        tools=[DuckDuckGo()],
        markdown=True,
    )

##Initializing the agent

multimodal_agent=initializer_agent()

video_file=st.file_uploader(
    "Upload your video  ", type=['mp4','mov','avi'], help="Upload a video for AI analysis"
)

if video_file:
    with tempfile.NamedTemporaryFile(delete=False,suffix='.mp4') as temp_video:
        temp_video.write(video_file.read())
        video_path=temp_video.name

    st.video(video_path,format='video/mp4',start_time=0)

    user_query= st.text_area(
        "What insights are you seeking from the video?",
        placeholder="Ask anything about the video, the AI will analyse it and provide you an answer.",
        help="Provide specific questions or insights about the video"
    )

    if st.button("Analyze video",key="Analyse_video_button"):
        if not user_query:
            st.warning("Please enter a question or insight to analyze the video.")
        else:
            try:
                with st.spinner("Processing video and gathering information"):
                    processed_video=upload_file(video_path)
                    while processed_video.state.name=="PROCESSING":
                        time.sleep(1)
                        processed_video=get_file(processed_video.name)

                    analysis_prompt=(
                        f"""
                        Analyze the uploaded video for content and context. Respond the following query using video insights and
                        supplementry web research
                        {user_query}
                        Provide a detailed, user-friendly and actionable response
                        """
                    )

                    response=multimodal_agent.run(analysis_prompt,videos=[processed_video])

                st.subheader(" Analysing RESULT")
                st.markdown(response.content)

            except Exception as error:
                st.error(f"An error occured during analysis: {error}")
            finally:
                Path(video_path).unlink(missing_ok=True)
else:
    st.info("Upload a video file to begin ")

st.markdown(
    """
    <style>
    .stTextArea textarea(
        height: 100px;
        )
    </style>

    """,
    unsafe_allow_html=True
)
