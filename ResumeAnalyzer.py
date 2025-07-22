import streamlit as st
import PyPDF2
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import io


# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')


download_nltk_data()


class ResumeAnalyzer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    def extract_text_from_pdf(self, pdf_file):
        """Extract text from uploaded PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return ""

    def preprocess_text(self, text):
        """Clean and preprocess text"""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stopwords
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
        return ' '.join(tokens)

    def extract_skills(self, text):
        """Extract potential skills from text"""
        # Common technical skills (you can expand this list)
        tech_skills = {
            'python', 'java', 'javascript', 'react', 'nodejs', 'html', 'css', 'sql',
            'mysql', 'postgresql', 'mongodb', 'django', 'flask', 'streamlit', 'pandas',
            'numpy', 'tensorflow', 'pytorch', 'scikit-learn', 'machine learning', 'ai',
            'data science', 'nlp', 'deep learning', 'aws', 'azure', 'docker', 'kubernetes',
            'git', 'github', 'linux', 'agile', 'scrum', 'api', 'rest', 'microservices',
            'cloud computing', 'devops', 'ci/cd', 'tableau', 'powerbi', 'excel'
        }

        text_lower = text.lower()
        found_skills = []

        for skill in tech_skills:
            if skill in text_lower:
                found_skills.append(skill)

        return found_skills

    def calculate_similarity(self, resume_text, job_description):
        """Calculate similarity between resume and job description"""
        # Preprocess both texts
        resume_processed = self.preprocess_text(resume_text)
        job_processed = self.preprocess_text(job_description)

        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform([resume_processed, job_processed])

        # Calculate cosine similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

        return similarity * 100  # Convert to percentage

    def extract_keywords(self, text, top_n=20):
        """Extract top keywords from text"""
        processed_text = self.preprocess_text(text)

        # Use TF-IDF to find important words
        vectorizer = TfidfVectorizer(max_features=top_n, ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform([processed_text])

        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.toarray()[0]

        # Create keyword-score pairs
        keywords = list(zip(feature_names, tfidf_scores))
        keywords.sort(key=lambda x: x[1], reverse=True)

        return keywords

    def match_keywords(self, resume_keywords, job_keywords):
        """Find matching keywords between resume and job description"""
        resume_words = set([kw[0] for kw in resume_keywords])
        job_words = set([kw[0] for kw in job_keywords])

        matched = resume_words.intersection(job_words)
        missing = job_words - resume_words

        return list(matched), list(missing)

    def generate_recommendations(self, missing_keywords, resume_skills, job_skills):
        """Generate improvement recommendations"""
        recommendations = []

        if missing_keywords:
            recommendations.append(f"📝 Consider adding these keywords: {', '.join(missing_keywords[:10])}")

        missing_skills = set(job_skills) - set(resume_skills)
        if missing_skills:
            recommendations.append(f"🔧 Develop these skills: {', '.join(list(missing_skills)[:5])}")

        if len(resume_skills) < 5:
            recommendations.append("💡 Add more technical skills to your resume")

        if not recommendations:
            recommendations.append("✅ Your resume looks well-aligned with the job description!")

        return recommendations


def main():
    st.set_page_config(
        page_title="AI Resume Analyzer",
        page_icon="📄",
        layout="wide"
    )

    st.title("🎯 AI-Powered Resume Analyzer")
    st.markdown("*Analyze your resume against job descriptions and get personalized recommendations*")

    # Initialize analyzer
    analyzer = ResumeAnalyzer()

    # Sidebar
    st.sidebar.header("📋 Instructions")
    st.sidebar.markdown("""
    1. Upload your resume (PDF format)
    2. Paste the job description
    3. Click 'Analyze Resume'
    4. Review your compatibility score and recommendations
    """)

    # Main interface
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📄 Upload Resume")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload your resume in PDF format"
        )

        if uploaded_file:
            st.success("✅ Resume uploaded successfully!")

    with col2:
        st.header("📝 Job Description")
        job_description = st.text_area(
            "Paste the job description here:",
            height=200,
            placeholder="Copy and paste the complete job description..."
        )

    if st.button("🔍 Analyze Resume", type="primary"):
        if uploaded_file and job_description:
            with st.spinner("Analyzing your resume..."):
                # Extract text from PDF
                resume_text = analyzer.extract_text_from_pdf(uploaded_file)

                if resume_text:
                    # Calculate similarity score
                    similarity_score = analyzer.calculate_similarity(resume_text, job_description)

                    # Extract skills
                    resume_skills = analyzer.extract_skills(resume_text)
                    job_skills = analyzer.extract_skills(job_description)

                    # Extract keywords
                    resume_keywords = analyzer.extract_keywords(resume_text)
                    job_keywords = analyzer.extract_keywords(job_description)

                    # Match keywords
                    matched_keywords, missing_keywords = analyzer.match_keywords(
                        resume_keywords, job_keywords
                    )

                    # Generate recommendations
                    recommendations = analyzer.generate_recommendations(
                        missing_keywords, resume_skills, job_skills
                    )

                    # Display results
                    st.header("📊 Analysis Results")

                    # Similarity score with gauge
                    col1, col2, col3 = st.columns([2, 1, 1])

                    with col1:
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number+delta",
                            value=similarity_score,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Resume Match Score"},
                            gauge={
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 50], 'color': "lightgray"},
                                    {'range': [50, 80], 'color': "yellow"},
                                    {'range': [80, 100], 'color': "lightgreen"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 90
                                }
                            }
                        ))
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        st.metric("Skills Found", len(resume_skills))
                        st.metric("Keywords Matched", len(matched_keywords))

                    with col3:
                        st.metric("Skills Required", len(job_skills))
                        st.metric("Missing Keywords", len(missing_keywords))

                    # Detailed breakdown
                    tab1, tab2, tab3, tab4 = st.tabs(
                        ["🎯 Recommendations", "🔧 Skills Analysis", "🔤 Keywords", "📈 Detailed Metrics"])

                    with tab1:
                        st.subheader("💡 Recommendations")
                        for rec in recommendations:
                            st.write(rec)

                        # Match percentage interpretation
                        if similarity_score >= 80:
                            st.success("🎉 Excellent match! Your resume is well-aligned with this position.")
                        elif similarity_score >= 60:
                            st.warning("⚠️ Good match with room for improvement. Consider the recommendations above.")
                        else:
                            st.error("❗ Low match. Significant improvements needed to align with this position.")

                    with tab2:
                        col1, col2 = st.columns(2)

                        with col1:
                            st.subheader("Your Skills")
                            if resume_skills:
                                for skill in resume_skills[:10]:
                                    st.write(f"✅ {skill}")
                            else:
                                st.write("No technical skills detected")

                        with col2:
                            st.subheader("Required Skills")
                            if job_skills:
                                for skill in job_skills[:10]:
                                    if skill in resume_skills:
                                        st.write(f"✅ {skill}")
                                    else:
                                        st.write(f"❌ {skill}")

                    with tab3:
                        col1, col2 = st.columns(2)

                        with col1:
                            st.subheader("Matched Keywords")
                            if matched_keywords:
                                df_matched = pd.DataFrame(matched_keywords[:15], columns=['Keyword'])
                                st.dataframe(df_matched, use_container_width=True)
                            else:
                                st.write("No matching keywords found")

                        with col2:
                            st.subheader("Missing Keywords")
                            if missing_keywords:
                                df_missing = pd.DataFrame(missing_keywords[:15], columns=['Keyword'])
                                st.dataframe(df_missing, use_container_width=True)
                            else:
                                st.write("No missing keywords")

                    with tab4:
                        # Create comparison charts
                        metrics_data = {
                            'Metric': ['Skills Match', 'Keyword Match', 'Overall Score'],
                            'Percentage': [
                                (len(set(resume_skills).intersection(set(job_skills))) / max(len(job_skills), 1)) * 100,
                                (len(matched_keywords) / max(len(job_keywords), 1)) * 100,
                                similarity_score
                            ]
                        }

                        df_metrics = pd.DataFrame(metrics_data)

                        fig = px.bar(
                            df_metrics,
                            x='Metric',
                            y='Percentage',
                            title='Resume Analysis Breakdown',
                            color='Percentage',
                            color_continuous_scale='RdYlGn'
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Could not extract text from the PDF. Please ensure it's a valid PDF file.")
        else:
            st.warning("Please upload a resume and enter a job description.")

    # Footer
    st.markdown("---")
    st.markdown("*Built with Python, Streamlit, and NLP • Resume Analyzer v2.0*")


if __name__ == "__main__":
    main()