import io
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import librosa
from transformers import pipeline

st.set_page_config(page_title='Speech Emotion Recognition', page_icon='🎙️', layout='wide')
st.title('🎙️ Speech Emotion Recognition')
st.write('Upload a speech recording and the AI model will estimate the emotion expressed in the audio.')

@st.cache_resource
def load_model():
    return pipeline('audio-classification', model='superb/wav2vec2-base-superb-er')

@st.cache_data
def load_audio(audio_bytes):
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    rms = float(np.mean(librosa.feature.rms(y=y)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    return y, sr, duration, rms, zcr

uploaded_file = st.file_uploader('Upload a speech recording', type=['wav','mp3','ogg','flac','m4a'])

if uploaded_file:
    audio_bytes = uploaded_file.getvalue()
    st.audio(audio_bytes, format=uploaded_file.type)
    try:
        y, sr, duration, rms, zcr = load_audio(audio_bytes)
    except Exception as e:
        st.error(f'Could not process the audio: {e}')
        st.stop()

    c1, c2, c3 = st.columns(3)
    c1.metric('Duration', f'{duration:.2f} sec')
    c2.metric('Sample Rate', f'{sr} Hz')
    c3.metric('Audio Energy', f'{rms:.4f}')

    if st.button('🧠 Analyze Emotion', use_container_width=True):
        with st.spinner('Analyzing speech emotion...'):
            try:
                classifier = load_model()
                result = classifier({'raw': y, 'sampling_rate': sr})
            except Exception as e:
                st.error(f'The AI model could not analyze this audio. Details: {e}')
                st.stop()

        result = sorted(result, key=lambda x: x['score'], reverse=True)
        top = result[0]
        label = top['label']
        confidence = top['score']

        st.markdown('---')
        st.subheader('🎯 Prediction')
        st.success(f'Predicted Emotion: **{label.upper()}**')
        st.metric('Confidence', f'{confidence * 100:.2f}%')

        if confidence < 0.60:
            st.warning('The model has relatively low confidence. Treat this prediction as an estimate.')
        elif confidence < 0.80:
            st.info('The model has moderate confidence in this prediction.')
        else:
            st.success('The model has high confidence in this prediction.')

        st.subheader('📊 Emotion Probabilities')
        labels = [x['label'] for x in result]
        scores = [x['score'] for x in result]
        fig, ax = plt.subplots()
        ax.bar(labels, scores)
        ax.set_ylim(0, 1)
        ax.set_ylabel('Probability')
        ax.set_xlabel('Emotion')
        ax.set_title('Predicted Emotion Probabilities')
        plt.xticks(rotation=30, ha='right')
        st.pyplot(fig)

        st.subheader('🔊 Basic Audio Features')
        a, b = st.columns(2)
        a.write(f'**RMS Energy:** {rms:.4f}')
        b.write(f'**Zero Crossing Rate:** {zcr:.4f}')

        with st.expander('ℹ️ About this project'):
            st.write('''This project uses a pretrained Wav2Vec2 speech emotion recognition model. The application accepts speech audio, processes the waveform, and predicts the emotion expressed in the recording.\n\nTechnologies: Python, Streamlit, Librosa, PyTorch, Hugging Face Transformers, Wav2Vec2.\n\nImportant: Emotion recognition is an AI estimate based on acoustic and learned speech patterns. It should not be treated as a definitive measurement of a person's emotional state.''')
else:
    st.info('👆 Upload a speech recording above to start emotion analysis.')
