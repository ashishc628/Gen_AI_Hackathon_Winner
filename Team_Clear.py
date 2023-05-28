import os
import streamlit as st
import sounddevice as sd
import soundfile as sf
import openai
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper
import os
import streamlit as st
import sounddevice as sd
import soundfile as sf
import openai
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as waves
import csv
import pandas as pd
from scipy import signal
from scipy import stats 
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff

# Set your OpenAI API key
os.environ['OPENAI_API_KEY'] = "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"

st.image("/Users/ashishchaudhary/Desktop/teamclear/Brown Blue Denim Jeans Fashion Sale Website Banner.jpg", use_column_width=True)
st.header('Voice Analysis')


# App framework
#st.title('Clear - Your Mental Health Tracker 🧠')
st.caption('Clear employs Cognitive Behavioral Therapy (CBT) modules to interpret user entries, identifying disorders in their speech and offering effective countermeasures. It mirrors your tone and language, and suggests mood-appropriate podcasts, songs, and more.')
prompt = st.text_input('How are you feeling today? Share your rawest emotions!')


# Prompt templates
title_template = PromptTemplate(
    input_variables=['entry'],
    template='''
        As a revered and witty Behavioral Psychology expert, analyze a given journal entry:
        Pinpoint these aspects in 2-4 words, even with insufficient data. Be an entertaining companion:
        Display all number headings in Bold.

        "Your Mood Card"

        1. Mood

        2. Trigger

        3. Focus

        4. Personality

        5. Mental profile

        6. Environment

        7. Habit

        8. Identified Distortion: Mention and explain the identified cognitive distortion

        9. Your Analysis: (20-40 words) Subtly identify cognitive distortions (like all-or-nothing thinking, mind-reading, personalization, should-ing and must-ing, mental filter, overgeneralization, magnification, minimization, fortune-telling, comparison, catastrophizing, labeling, disqualifying the positive) but don't mention them directly.

        "Recommendations for you"

        1. Personalized Solution: Offer humorous, inventive guidance and immediate steps for improvement in 20-50 words, countering. Don't use formal addresses.
        2. Song recommendation echoing user's emotions with working and latest links
        3. Podcast: Recommend a fitting podcast episode with working and latest links and explain its relevance to user's current mood. Avoid repetition.
        4. Meditation that will help: Share a working YouTube link to a relevant meditation that will uplift the user.
        5. Movie: Recommend a fitting movie with working and latest trailer links and explain its relevance to user's current mood. Avoid repetition.

        See you soon! Craft a personalized, amusing 'see you again' message (20-30 words) for an eager return. {entry}
    '''
)

# Memory
title_memory = ConversationBufferMemory(input_key='entry', memory_key='chat_history')

# Llms
llm = ChatOpenAI(model_name="gpt-4", temperature=0.7)
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='entry', memory=title_memory)

# Streamlit app title and file upload
#st.title("Whisper Audio Processing")

# Start audio recording button
if st.button("Start Audio Recording"):
    st.write("Audio recording started...")

    # Placeholder for storing audio data
    audio_data = []

    # Sample rate and duration for recording
    sample_rate = 44100
    duration = 10  # Recording duration in seconds

    # Start recording audio
    audio_data = sd.rec(int(sample_rate * duration), samplerate=sample_rate, channels=1)
    sd.wait()  # Wait for the recording to complete

    # Save the recorded audio to a file
    audio_file_path = "recorded_audio.wav"
    sf.write(audio_file_path, audio_data, sample_rate)

    st.write("Audio saved!")

    # Play the recorded audio
    st.audio(audio_file_path, format="audio/wav")

    # Audio transcription
    st.write("Transcribing audio...")
    try:
        # Load the audio file as a file-like object
        with open(audio_file_path, "rb") as audio_file:
            transcript = openai.Audio.transcribe("whisper-1", audio_file)
        st.write("Transcription:")
        st.write(transcript)
    except openai.OpenAIError as e:
        st.error(f"Error occurred during transcription: {str(e)}")

    # Audio translation
    st.write("Translating audio...")
    try:
        # Load the audio file as a file-like object
        with open(audio_file_path, "rb") as audio_file:
            translation = openai.Audio.translate("whisper-1", audio_file)
        st.write("Translation:")
        st.write(translation)
    except openai.OpenAIError as e:
        st.error(f"Error occurred during translation: {str(e)}")

    # Generate LangChain prompt using the transcription or translation
    if prompt and (transcript or translation):
        entry = transcript if transcript is not None else translation
        title = title_chain.run(entry=entry)
        st.write(title)

# Text input prompt
if prompt:
    title = title_chain.run(entry=prompt)
    st.write(title)

import cv2
import streamlit as st
import tempfile
import shutil
import os
import time

st.header('Facial Emotion Analysis')
st.caption('Clear uses an API call from an AI toolkit to measure, understand, and improve emotional expression.')
st.caption("Talk about something you love.")

# Checkbox to start/stop capturing frames
run = st.checkbox('Start Video Recording')
download_video = False

# Create a container for the download button
button_container = st.container()

if run:
    # Initialize video capture
    camera = cv2.VideoCapture(0)

    # Create a temporary directory to store the frames
    temp_dir = tempfile.mkdtemp()
    output_directory = os.path.join(temp_dir, "video_frames")
    os.makedirs(output_directory, exist_ok=True)

    # Create a video writer to save the frames as a video file
    fps = camera.get(cv2.CAP_PROP_FPS)
    video_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_path = os.path.join(temp_dir, "live_video.mp4")
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (video_width, video_height))

    # Create a placeholder for displaying frames
    frame_placeholder = st.empty()

    start_time = time.time()
    while run:
        _, frame = camera.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_writer.write(frame)

        if time.time() - start_time >= 15:
            download_video = True
            break

        # Display the frame
        frame_placeholder.image(frame)

    # Release the video writer and camera
    video_writer.release()
    camera.release()

    if download_video:
        # Download the video file
        with open(video_path, "rb") as file:
            st.download_button("Download Video", data=file, file_name="live_video.mp4", mime="video/mp4")

        # Clean up the temporary directory
        shutil.rmtree(temp_dir)

if button_container.button("Run"):
    if not run:
        download_video = True
    run = not run

# Set up Streamlit layout

# Upload CSV file
uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file is not None:
    # Read the uploaded CSV file
    df = pd.read_csv(uploaded_file)

    # Sort DataFrame by top 10 descending order of emotion scores
    df_sorted = df.sort_values('emotion_score', ascending=False).head(10)

    # Doughnut chart
    st.subheader('Doughnut Chart')
    doughnut_data = df_sorted['emotion_score']
    labels = df_sorted['emotion_name']
    total = doughnut_data.sum()
    percentages = [f"{score/total*100:.1f}%" for score in doughnut_data]
    doughnut_fig = px.pie(df_sorted, values=doughnut_data, names=labels, hole=0.5, labels=percentages)
    st.plotly_chart(doughnut_fig)

    # Area chart
    st.subheader('Area Chart')
    area_fig = px.area(df_sorted, x='emotion_name', y='emotion_score')
    st.plotly_chart(area_fig)

    # Bar chart
    st.subheader('Bar Chart')
    bar_fig = px.bar(df_sorted, x='emotion_name', y='emotion_score')
    st.plotly_chart(bar_fig)

    # Cluster graph
    st.subheader('Cluster Graph')
    cluster_fig = px.scatter(df_sorted, x='emotion_name', y='emotion_score', color='emotion_name')
    st.plotly_chart(cluster_fig)

    # Treemap
    st.subheader('Treemap')
    treemap_fig = px.treemap(df_sorted, path=['emotion_name'], values='emotion_score', color='emotion_score', hover_data=['emotion_name', 'emotion_score'])
    st.plotly_chart(treemap_fig)


def smoothTriangle(data, degree):
    triangle = np.concatenate((np.arange(degree + 1), np.arange(degree)[::-1]))  # up then down
    smoothed = []

    for i in range(degree, len(data) - degree * 2):
        point = data[i:i + len(triangle)] * triangle
        smoothed.append(np.sum(point) / np.sum(triangle))
    
    # Handle boundaries
    smoothed = [smoothed[0]] * int(degree + degree / 2) + smoothed
    while len(smoothed) < len(data):
        smoothed.append(smoothed[-1])
    
    return smoothed

def main():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.header("Brain Wave  Analysis")
    st.caption('Clear harnesses these EEG patterns to enhance your understanding of mental wellness, facilitating improved focus, calmness, and cognitive activity.')

    file = st.file_uploader("Upload WAV File", type=["wav"])
    
    if file is not None:
        fs, data = waves.read(file)

        length_data = np.shape(data)
        length_new = length_data[0] * 0.05
        ld_int = int(length_new)

        data_new = signal.resample(data, ld_int)

        fig_spectrogram = plt.figure('Spectrogram')
        d, f, t, im = plt.specgram(data_new, NFFT=256, Fs=500, noverlap=250)
        plt.ylim(0, 90)
        plt.colorbar(label="Power/Frequency")
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [s]')
        st.pyplot(fig_spectrogram)

        matrixf = np.array(f).T
        np.savetxt('Frequencies.csv', matrixf)
        df = pd.read_csv("Frequencies.csv", header=None, index_col=None)
        df.columns = ["Frequencies"]
        df.to_csv("Frequencies.csv", index=False)

        position_vector = []
        length_f = np.shape(f)
        l_row_f = length_f[0]
        for i in range(0, l_row_f):
            if f[i] >= 7 and f[i] <= 12:
                position_vector.append(i)

        length_d = np.shape(d)
        l_col_d = length_d[1] if len(length_d) > 1 else 0
        AlphaRange = [np.mean(d[position_vector[0]:max(position_vector) + 1, i]) for i in range(l_col_d)]

        fig_alpha_range = plt.figure('AlphaRange')
        y = smoothTriangle(AlphaRange, 100)
        plt.plot(t, y)
        plt.xlabel('Time [s]')
        plt.xlim(0, max(t))
        st.pyplot(fig_alpha_range)

        datosy = np.asarray(y)
        datosyt = np.array([datosy, t])
        with open('datosyt.csv', 'w', newline='') as file:
            writer = csv.writer(file, dialect='excel-tab')
            writer.writerows(datosyt.T)

        df = pd.read_csv("datosyt.csv", header=None, index_col=None)
        df.columns = ["Power"]
        df.to_csv("datosyt.csv", index=False)

        tg = np.array([4.2552, 14.9426, 23.2801, 36.0951, 45.4738, 59.3751, 72.0337, 85.0831, max(t) + 1])

        length_t = np.shape(t)
        l_row_t = length_t[0]
        eyesopen = []
        eyesclosed = []
        j = 0  # initial variable to traverse tg
        l = 0  # initial variable to loop through the "y" data
        for i in range(0, l_row_t):
            if t[i] >= tg[j]:
                if j % 2 == 0:
                    eyesopen.append(np.mean(datosy[l:i]))
                if j % 2 == 1:
                    eyesclosed.append(np.mean(datosy[l:i]))
                l = i
                j = j + 1

        fig_data_analysis = plt.figure('DataAnalysis')
        plt.boxplot([eyesopen, eyesclosed], sym='ko', whis=1.5)
        plt.xticks([1, 2], ['Eyes open', 'Eyes closed'], size='small', color='k')
        plt.ylabel('AlphaPower')
        st.pyplot(fig_data_analysis)

        meanopen = np.mean(eyesopen)
        meanclosed = np.mean(eyesclosed)
        sdopen = np.std(eyesopen)
        sdclosed = np.std(eyesclosed)
        eyes = np.array([eyesopen, eyesclosed])

        result = stats.ttest_ind(eyesopen, eyesclosed, equal_var=False)

        st.subheader("Results")
        st.write("Mean (Eyes Open):", meanopen)
        st.write("Mean (Eyes Closed):", meanclosed)
        st.write("Standard Deviation (Eyes Open):", sdopen)
        st.write("Standard Deviation (Eyes Closed):", sdclosed)
        st.write("T-Test Result:", result)

if __name__ == "__main__":
    main()